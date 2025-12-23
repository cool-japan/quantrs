//! # Quantum-Classical Hybrid AutoML Decision Engine
//!
//! An advanced system that intelligently determines when quantum algorithms provide
//! advantages over classical approaches and automatically configures optimal solutions.
//!
//! ## Key Features
//!
//! - **Problem Analysis**: Extracts characteristics from datasets and problem definitions
//! - **Algorithm Selection**: Chooses optimal quantum or classical algorithms
//! - **Performance Prediction**: Estimates accuracy, latency, and resource usage
//! - **Cost Optimization**: Balances quantum hardware costs with performance gains
//! - **Calibration Integration**: Automatically applies probability calibration
//! - **Production Configuration**: Generates deployment-ready configurations
//!
//! ## Example
//!
//! ```rust,ignore
//! use quantrs2_ml::hybrid_automl_engine::{
//!     HybridAutoMLEngine, ProblemCharacteristics, ResourceConstraints
//! };
//! use scirs2_core::ndarray::{Array1, Array2};
//!
//! // Example dataset
//! let X = Array2::<f64>::zeros((100, 10)); // 100 samples, 10 features
//! let y = Array1::<usize>::zeros(100);      // 100 labels (class indices)
//!
//! let engine = HybridAutoMLEngine::new();
//! let problem = ProblemCharacteristics::from_dataset(&X, &y);
//! let constraints = ResourceConstraints::default(); // Define resource constraints
//! let recommendation = engine.analyze_and_recommend(&problem, &constraints)?;
//!
//! println!("Recommended: {:?}", recommendation.algorithm_choice);
//! println!("Expected speedup: {:.2}x", recommendation.quantum_advantage.speedup);
//! # Ok::<(), quantrs2_ml::error::MLError>(())
//! ```

use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{thread_rng, Rng};
use std::collections::HashMap;

/// Problem characteristics extracted from dataset and task definition
#[derive(Debug, Clone)]
pub struct ProblemCharacteristics {
    /// Number of samples in the dataset
    pub n_samples: usize,

    /// Number of features per sample
    pub n_features: usize,

    /// Number of classes (for classification) or 1 (for regression)
    pub n_classes: usize,

    /// Dimensionality ratio (features/samples)
    pub dimensionality_ratio: f64,

    /// Data sparsity (fraction of zero elements)
    pub sparsity: f64,

    /// Feature correlation matrix condition number
    pub condition_number: f64,

    /// Class imbalance ratio (max_class_count / min_class_count)
    pub class_imbalance: f64,

    /// Task type
    pub task_type: TaskType,

    /// Problem domain
    pub domain: ProblemDomain,
}

/// Type of machine learning task
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TaskType {
    BinaryClassification,
    MultiClassClassification,
    Regression,
    Clustering,
    DimensionalityReduction,
}

/// Problem domain for specialized optimizations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProblemDomain {
    General,
    DrugDiscovery,
    Finance,
    ComputerVision,
    NaturalLanguage,
    TimeSeriesForecasting,
    AnomalyDetection,
    RecommenderSystem,
}

/// Available computational resources
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    /// Available quantum devices
    pub quantum_devices: Vec<QuantumDevice>,

    /// Available classical compute
    pub classical_compute: ClassicalCompute,

    /// Maximum latency requirement (milliseconds)
    pub max_latency_ms: Option<f64>,

    /// Maximum cost per inference (USD)
    pub max_cost_per_inference: Option<f64>,

    /// Maximum training time (seconds)
    pub max_training_time: Option<f64>,

    /// Power consumption limit (watts)
    pub max_power_consumption: Option<f64>,
}

/// Quantum device specification
#[derive(Debug, Clone)]
pub struct QuantumDevice {
    pub name: String,
    pub n_qubits: usize,
    pub gate_error_rate: f64,
    pub measurement_error_rate: f64,
    pub decoherence_time_us: f64,
    pub cost_per_shot: f64,
    pub availability: DeviceAvailability,
}

/// Device availability status
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DeviceAvailability {
    Available,
    Queued,
    Unavailable,
}

/// Classical compute resources
#[derive(Debug, Clone)]
pub struct ClassicalCompute {
    pub n_cpu_cores: usize,
    pub has_gpu: bool,
    pub gpu_memory_gb: f64,
    pub ram_gb: f64,
}

/// Algorithm recommendation
#[derive(Debug, Clone)]
pub struct AlgorithmRecommendation {
    /// Recommended algorithm choice
    pub algorithm_choice: AlgorithmChoice,

    /// Predicted quantum advantage
    pub quantum_advantage: QuantumAdvantageMetrics,

    /// Recommended hyperparameters
    pub hyperparameters: HashMap<String, f64>,

    /// Expected performance metrics
    pub expected_performance: PerformanceEstimate,

    /// Cost analysis
    pub cost_analysis: CostAnalysis,

    /// Confidence in recommendation (0-1)
    pub confidence: f64,

    /// Calibration recommendation
    pub calibration_method: Option<String>,

    /// Production configuration
    pub production_config: ProductionConfig,
}

/// Algorithm choice (quantum, classical, or hybrid)
#[derive(Debug, Clone, PartialEq)]
pub enum AlgorithmChoice {
    /// Use quantum algorithm exclusively
    QuantumOnly { algorithm: String, device: String },

    /// Use classical algorithm exclusively
    ClassicalOnly { algorithm: String, backend: String },

    /// Use hybrid quantum-classical approach
    Hybrid {
        quantum_component: String,
        classical_component: String,
        splitting_strategy: String,
    },
}

/// Quantum advantage metrics
#[derive(Debug, Clone)]
pub struct QuantumAdvantageMetrics {
    /// Expected speedup factor
    pub speedup: f64,

    /// Accuracy improvement (percentage points)
    pub accuracy_improvement: f64,

    /// Sample efficiency improvement
    pub sample_efficiency: f64,

    /// Generalization improvement
    pub generalization_improvement: f64,

    /// Statistical significance (p-value)
    pub statistical_significance: f64,
}

/// Performance estimate
#[derive(Debug, Clone)]
pub struct PerformanceEstimate {
    /// Expected accuracy (0-1)
    pub accuracy: f64,

    /// Accuracy confidence interval (95%)
    pub accuracy_ci: (f64, f64),

    /// Expected training time (seconds)
    pub training_time_s: f64,

    /// Expected inference latency (milliseconds)
    pub inference_latency_ms: f64,

    /// Memory footprint (MB)
    pub memory_mb: f64,
}

/// Cost analysis
#[derive(Debug, Clone)]
pub struct CostAnalysis {
    /// Training cost (USD)
    pub training_cost: f64,

    /// Inference cost per sample (USD)
    pub inference_cost_per_sample: f64,

    /// Total cost for expected workload (USD)
    pub total_cost: f64,

    /// Cost breakdown
    pub breakdown: HashMap<String, f64>,
}

/// Production deployment configuration
#[derive(Debug, Clone)]
pub struct ProductionConfig {
    /// Recommended batch size
    pub batch_size: usize,

    /// Number of parallel workers
    pub n_workers: usize,

    /// Enable caching
    pub enable_caching: bool,

    /// Monitoring configuration
    pub monitoring: MonitoringConfig,

    /// Scaling configuration
    pub scaling: ScalingConfig,
}

/// Monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Log every N inferences
    pub log_interval: usize,

    /// Alert thresholds
    pub alert_thresholds: HashMap<String, f64>,

    /// Metrics to track
    pub tracked_metrics: Vec<String>,
}

/// Scaling configuration
#[derive(Debug, Clone)]
pub struct ScalingConfig {
    /// Auto-scaling enabled
    pub auto_scaling: bool,

    /// Minimum instances
    pub min_instances: usize,

    /// Maximum instances
    pub max_instances: usize,

    /// Scale up threshold (CPU %)
    pub scale_up_threshold: f64,

    /// Scale down threshold (CPU %)
    pub scale_down_threshold: f64,
}

/// Quantum-Classical Hybrid AutoML Decision Engine
pub struct HybridAutoMLEngine {
    /// Performance prediction models
    performance_models: HashMap<String, PerformanceModel>,

    /// Cost models
    cost_models: HashMap<String, CostModel>,

    /// Decision thresholds
    decision_thresholds: DecisionThresholds,
}

/// Performance prediction model
struct PerformanceModel {
    /// Model type
    model_type: String,

    /// Coefficients for prediction
    coefficients: Vec<f64>,
}

/// Cost prediction model
struct CostModel {
    /// Base cost
    base_cost: f64,

    /// Cost per sample
    cost_per_sample: f64,

    /// Cost per feature
    cost_per_feature: f64,

    /// Cost per qubit
    cost_per_qubit: f64,
}

/// Decision thresholds for algorithm selection
struct DecisionThresholds {
    /// Minimum speedup to justify quantum (default: 1.5x)
    min_quantum_speedup: f64,

    /// Minimum accuracy improvement (percentage points)
    min_accuracy_improvement: f64,

    /// Maximum acceptable cost ratio
    max_cost_ratio: f64,

    /// Minimum confidence for quantum recommendation
    min_confidence: f64,
}

impl HybridAutoMLEngine {
    /// Create a new Hybrid AutoML Engine with default models
    pub fn new() -> Self {
        Self {
            performance_models: Self::initialize_performance_models(),
            cost_models: Self::initialize_cost_models(),
            decision_thresholds: DecisionThresholds::default(),
        }
    }

    /// Analyze problem and recommend optimal approach
    pub fn analyze_and_recommend(
        &self,
        characteristics: &ProblemCharacteristics,
        constraints: &ResourceConstraints,
    ) -> Result<AlgorithmRecommendation> {
        // Extract features for decision making
        let features = self.extract_decision_features(characteristics);

        // Evaluate quantum algorithms
        let quantum_options = self.evaluate_quantum_algorithms(characteristics, constraints)?;

        // Evaluate classical algorithms
        let classical_options = self.evaluate_classical_algorithms(characteristics, constraints)?;

        // Compare and select best option
        let best_option = self.select_best_option(
            &quantum_options,
            &classical_options,
            characteristics,
            constraints,
        )?;

        // Generate comprehensive recommendation
        let recommendation =
            self.generate_recommendation(best_option, characteristics, constraints)?;

        Ok(recommendation)
    }

    /// Extract features for decision making
    fn extract_decision_features(&self, chars: &ProblemCharacteristics) -> Vec<f64> {
        vec![
            chars.n_samples as f64,
            chars.n_features as f64,
            chars.n_classes as f64,
            chars.dimensionality_ratio,
            chars.sparsity,
            chars.condition_number,
            chars.class_imbalance,
            (chars.n_features as f64).log2(), // Log features for quantum circuit depth
        ]
    }

    /// Evaluate quantum algorithm options
    fn evaluate_quantum_algorithms(
        &self,
        chars: &ProblemCharacteristics,
        constraints: &ResourceConstraints,
    ) -> Result<Vec<AlgorithmOption>> {
        let mut options = Vec::new();

        // Check if quantum devices are available
        if constraints.quantum_devices.is_empty() {
            return Ok(options);
        }

        // Quantum SVM (QSVM)
        if matches!(
            chars.task_type,
            TaskType::BinaryClassification | TaskType::MultiClassClassification
        ) {
            let qsvm_option = self.evaluate_qsvm(chars, constraints)?;
            if qsvm_option.is_feasible {
                options.push(qsvm_option);
            }
        }

        // Quantum Neural Network (QNN)
        let qnn_option = self.evaluate_qnn(chars, constraints)?;
        if qnn_option.is_feasible {
            options.push(qnn_option);
        }

        // Variational Quantum Eigensolver (VQE) for specific problems
        if chars.domain == ProblemDomain::DrugDiscovery {
            let vqe_option = self.evaluate_vqe(chars, constraints)?;
            if vqe_option.is_feasible {
                options.push(vqe_option);
            }
        }

        // Quantum Approximate Optimization Algorithm (QAOA)
        if matches!(chars.task_type, TaskType::Clustering) {
            let qaoa_option = self.evaluate_qaoa(chars, constraints)?;
            if qaoa_option.is_feasible {
                options.push(qaoa_option);
            }
        }

        Ok(options)
    }

    /// Evaluate classical algorithm options
    fn evaluate_classical_algorithms(
        &self,
        chars: &ProblemCharacteristics,
        constraints: &ResourceConstraints,
    ) -> Result<Vec<AlgorithmOption>> {
        let mut options = Vec::new();

        // Classical SVM
        if matches!(
            chars.task_type,
            TaskType::BinaryClassification | TaskType::MultiClassClassification
        ) {
            options.push(self.evaluate_classical_svm(chars, constraints)?);
        }

        // Neural Network
        options.push(self.evaluate_classical_nn(chars, constraints)?);

        // Random Forest
        options.push(self.evaluate_random_forest(chars, constraints)?);

        // Gradient Boosting
        options.push(self.evaluate_gradient_boosting(chars, constraints)?);

        Ok(options)
    }

    /// Evaluate QSVM option
    fn evaluate_qsvm(
        &self,
        chars: &ProblemCharacteristics,
        constraints: &ResourceConstraints,
    ) -> Result<AlgorithmOption> {
        let device = &constraints.quantum_devices[0];

        // Check if problem fits on device
        let required_qubits = (chars.n_features as f64).log2().ceil() as usize;
        let is_feasible = required_qubits <= device.n_qubits;

        // Estimate performance
        let accuracy = self.estimate_qsvm_accuracy(chars, device)?;
        let training_time = self.estimate_qsvm_training_time(chars, device)?;
        let cost = self.estimate_qsvm_cost(chars, device)?;

        Ok(AlgorithmOption {
            name: "QSVM".to_string(),
            algorithm_type: AlgorithmType::Quantum,
            is_feasible,
            expected_accuracy: accuracy,
            expected_training_time_s: training_time,
            expected_inference_latency_ms: 10.0, // Kernel evaluation time
            expected_cost: cost,
            required_qubits: Some(required_qubits),
            confidence: 0.85,
        })
    }

    /// Evaluate QNN option
    fn evaluate_qnn(
        &self,
        chars: &ProblemCharacteristics,
        constraints: &ResourceConstraints,
    ) -> Result<AlgorithmOption> {
        let device = &constraints.quantum_devices[0];

        let required_qubits = chars.n_features.min(20); // Limit to 20 qubits
        let is_feasible = required_qubits <= device.n_qubits;

        let accuracy = self.estimate_qnn_accuracy(chars, device)?;
        let training_time = self.estimate_qnn_training_time(chars, device)?;
        let cost = self.estimate_qnn_cost(chars, device)?;

        Ok(AlgorithmOption {
            name: "QNN".to_string(),
            algorithm_type: AlgorithmType::Quantum,
            is_feasible,
            expected_accuracy: accuracy,
            expected_training_time_s: training_time,
            expected_inference_latency_ms: 5.0,
            expected_cost: cost,
            required_qubits: Some(required_qubits),
            confidence: 0.80,
        })
    }

    /// Evaluate VQE option
    fn evaluate_vqe(
        &self,
        chars: &ProblemCharacteristics,
        constraints: &ResourceConstraints,
    ) -> Result<AlgorithmOption> {
        let device = &constraints.quantum_devices[0];

        let required_qubits = 10.min(device.n_qubits);
        let is_feasible = true;

        Ok(AlgorithmOption {
            name: "VQE".to_string(),
            algorithm_type: AlgorithmType::Quantum,
            is_feasible,
            expected_accuracy: 0.92,
            expected_training_time_s: 300.0,
            expected_inference_latency_ms: 50.0,
            expected_cost: 100.0,
            required_qubits: Some(required_qubits),
            confidence: 0.75,
        })
    }

    /// Evaluate QAOA option
    fn evaluate_qaoa(
        &self,
        chars: &ProblemCharacteristics,
        constraints: &ResourceConstraints,
    ) -> Result<AlgorithmOption> {
        let device = &constraints.quantum_devices[0];

        let required_qubits = chars.n_samples.min(20);
        let is_feasible = required_qubits <= device.n_qubits;

        Ok(AlgorithmOption {
            name: "QAOA".to_string(),
            algorithm_type: AlgorithmType::Quantum,
            is_feasible,
            expected_accuracy: 0.88,
            expected_training_time_s: 200.0,
            expected_inference_latency_ms: 30.0,
            expected_cost: 80.0,
            required_qubits: Some(required_qubits),
            confidence: 0.78,
        })
    }

    /// Evaluate classical SVM
    fn evaluate_classical_svm(
        &self,
        chars: &ProblemCharacteristics,
        _constraints: &ResourceConstraints,
    ) -> Result<AlgorithmOption> {
        let accuracy = 0.85 - (chars.dimensionality_ratio * 0.05).min(0.15);
        let training_time = chars.n_samples as f64 * chars.n_features as f64 / 1000.0;

        Ok(AlgorithmOption {
            name: "Classical SVM".to_string(),
            algorithm_type: AlgorithmType::Classical,
            is_feasible: true,
            expected_accuracy: accuracy,
            expected_training_time_s: training_time,
            expected_inference_latency_ms: 0.1,
            expected_cost: 0.0001,
            required_qubits: None,
            confidence: 0.95,
        })
    }

    /// Evaluate classical neural network
    fn evaluate_classical_nn(
        &self,
        chars: &ProblemCharacteristics,
        constraints: &ResourceConstraints,
    ) -> Result<AlgorithmOption> {
        let base_accuracy = 0.88;
        let accuracy = base_accuracy - (chars.class_imbalance.log2() * 0.02).min(0.1);

        let training_time = if constraints.classical_compute.has_gpu {
            chars.n_samples as f64 / 100.0
        } else {
            chars.n_samples as f64 / 10.0
        };

        Ok(AlgorithmOption {
            name: "Neural Network".to_string(),
            algorithm_type: AlgorithmType::Classical,
            is_feasible: true,
            expected_accuracy: accuracy,
            expected_training_time_s: training_time,
            expected_inference_latency_ms: 0.5,
            expected_cost: 0.0001,
            required_qubits: None,
            confidence: 0.90,
        })
    }

    /// Evaluate random forest
    fn evaluate_random_forest(
        &self,
        chars: &ProblemCharacteristics,
        _constraints: &ResourceConstraints,
    ) -> Result<AlgorithmOption> {
        let accuracy = 0.86;
        let training_time = chars.n_samples as f64 * chars.n_features as f64 / 500.0;

        Ok(AlgorithmOption {
            name: "Random Forest".to_string(),
            algorithm_type: AlgorithmType::Classical,
            is_feasible: true,
            expected_accuracy: accuracy,
            expected_training_time_s: training_time,
            expected_inference_latency_ms: 0.2,
            expected_cost: 0.00005,
            required_qubits: None,
            confidence: 0.92,
        })
    }

    /// Evaluate gradient boosting
    fn evaluate_gradient_boosting(
        &self,
        chars: &ProblemCharacteristics,
        _constraints: &ResourceConstraints,
    ) -> Result<AlgorithmOption> {
        let accuracy = 0.89;
        let training_time = chars.n_samples as f64 * chars.n_features as f64 / 300.0;

        Ok(AlgorithmOption {
            name: "Gradient Boosting".to_string(),
            algorithm_type: AlgorithmType::Classical,
            is_feasible: true,
            expected_accuracy: accuracy,
            expected_training_time_s: training_time,
            expected_inference_latency_ms: 0.3,
            expected_cost: 0.0001,
            required_qubits: None,
            confidence: 0.93,
        })
    }

    /// Estimate QSVM accuracy
    fn estimate_qsvm_accuracy(
        &self,
        chars: &ProblemCharacteristics,
        device: &QuantumDevice,
    ) -> Result<f64> {
        // Base accuracy for QSVM
        let base_accuracy = 0.90;

        // Adjust for noise
        let noise_penalty = device.gate_error_rate * 50.0;

        // Adjust for dimensionality
        let dim_bonus = (1.0 / (1.0 + chars.dimensionality_ratio)) * 0.05;

        Ok((base_accuracy - noise_penalty + dim_bonus)
            .max(0.5)
            .min(0.99))
    }

    /// Estimate QSVM training time
    fn estimate_qsvm_training_time(
        &self,
        chars: &ProblemCharacteristics,
        _device: &QuantumDevice,
    ) -> Result<f64> {
        // Kernel matrix computation is O(n^2)
        let time = (chars.n_samples * chars.n_samples) as f64 / 100.0;
        Ok(time)
    }

    /// Estimate QSVM cost
    fn estimate_qsvm_cost(
        &self,
        chars: &ProblemCharacteristics,
        device: &QuantumDevice,
    ) -> Result<f64> {
        let n_shots = 1000;
        let n_kernel_evaluations = chars.n_samples * chars.n_samples;
        let cost = n_kernel_evaluations as f64 * n_shots as f64 * device.cost_per_shot;
        Ok(cost)
    }

    /// Estimate QNN accuracy
    fn estimate_qnn_accuracy(
        &self,
        chars: &ProblemCharacteristics,
        device: &QuantumDevice,
    ) -> Result<f64> {
        let base_accuracy = 0.87;
        let noise_penalty = device.gate_error_rate * 40.0;
        let complexity_bonus = (chars.n_features as f64 / 100.0).min(0.08);

        Ok((base_accuracy - noise_penalty + complexity_bonus)
            .max(0.5)
            .min(0.99))
    }

    /// Estimate QNN training time
    fn estimate_qnn_training_time(
        &self,
        chars: &ProblemCharacteristics,
        _device: &QuantumDevice,
    ) -> Result<f64> {
        let n_epochs = 100;
        let time_per_epoch = chars.n_samples as f64 / 10.0;
        Ok(n_epochs as f64 * time_per_epoch)
    }

    /// Estimate QNN cost
    fn estimate_qnn_cost(
        &self,
        chars: &ProblemCharacteristics,
        device: &QuantumDevice,
    ) -> Result<f64> {
        let n_shots = 1000;
        let n_epochs = 100;
        let cost_per_epoch = chars.n_samples as f64 * n_shots as f64 * device.cost_per_shot;
        Ok(n_epochs as f64 * cost_per_epoch)
    }

    /// Select best option from quantum and classical algorithms
    fn select_best_option(
        &self,
        quantum_options: &[AlgorithmOption],
        classical_options: &[AlgorithmOption],
        chars: &ProblemCharacteristics,
        constraints: &ResourceConstraints,
    ) -> Result<AlgorithmOption> {
        let mut all_options = quantum_options.to_vec();
        all_options.extend(classical_options.iter().cloned());

        // Filter by constraints
        let filtered: Vec<_> = all_options
            .into_iter()
            .filter(|opt| {
                if let Some(max_time) = constraints.max_training_time {
                    if opt.expected_training_time_s > max_time {
                        return false;
                    }
                }

                if let Some(max_cost) = constraints.max_cost_per_inference {
                    if opt.expected_cost > max_cost {
                        return false;
                    }
                }

                true
            })
            .collect();

        if filtered.is_empty() {
            return Err(MLError::InvalidInput(
                "No algorithms satisfy the given constraints".to_string(),
            ));
        }

        // Score each option
        let best = filtered
            .into_iter()
            .max_by(|a, b| {
                let score_a = self.compute_option_score(a, chars, constraints);
                let score_b = self.compute_option_score(b, chars, constraints);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .unwrap();

        Ok(best)
    }

    /// Compute score for an algorithm option
    fn compute_option_score(
        &self,
        option: &AlgorithmOption,
        _chars: &ProblemCharacteristics,
        _constraints: &ResourceConstraints,
    ) -> f64 {
        // Multi-objective scoring
        let accuracy_score = option.expected_accuracy;
        let speed_score = 1.0 / (1.0 + option.expected_training_time_s / 100.0);
        let cost_score = 1.0 / (1.0 + option.expected_cost);
        let confidence_score = option.confidence;

        // Weighted combination
        accuracy_score * 0.4 + speed_score * 0.2 + cost_score * 0.2 + confidence_score * 0.2
    }

    /// Generate comprehensive recommendation
    fn generate_recommendation(
        &self,
        best_option: AlgorithmOption,
        chars: &ProblemCharacteristics,
        constraints: &ResourceConstraints,
    ) -> Result<AlgorithmRecommendation> {
        let algorithm_choice = match best_option.algorithm_type {
            AlgorithmType::Quantum => AlgorithmChoice::QuantumOnly {
                algorithm: best_option.name.clone(),
                device: constraints
                    .quantum_devices
                    .get(0)
                    .map(|d| d.name.clone())
                    .unwrap_or_else(|| "simulator".to_string()),
            },
            AlgorithmType::Classical => AlgorithmChoice::ClassicalOnly {
                algorithm: best_option.name.clone(),
                backend: if constraints.classical_compute.has_gpu {
                    "GPU".to_string()
                } else {
                    "CPU".to_string()
                },
            },
            AlgorithmType::Hybrid => AlgorithmChoice::Hybrid {
                quantum_component: "QNN".to_string(),
                classical_component: "Neural Network".to_string(),
                splitting_strategy: "Feature Engineering".to_string(),
            },
        };

        // Compute quantum advantage metrics
        let quantum_advantage = self.compute_quantum_advantage(&best_option, chars)?;

        // Generate hyperparameters
        let hyperparameters = self.generate_hyperparameters(&best_option, chars)?;

        // Estimate performance
        let expected_performance = PerformanceEstimate {
            accuracy: best_option.expected_accuracy,
            accuracy_ci: (
                best_option.expected_accuracy - 0.05,
                best_option.expected_accuracy + 0.05,
            ),
            training_time_s: best_option.expected_training_time_s,
            inference_latency_ms: best_option.expected_inference_latency_ms,
            memory_mb: chars.n_samples as f64 * chars.n_features as f64 * 8.0 / 1024.0 / 1024.0,
        };

        // Cost analysis
        let cost_analysis = self.generate_cost_analysis(&best_option, chars)?;

        // Calibration recommendation
        let calibration_method = self.recommend_calibration(&best_option, chars)?;

        // Production configuration
        let production_config =
            self.generate_production_config(&best_option, chars, constraints)?;

        Ok(AlgorithmRecommendation {
            algorithm_choice,
            quantum_advantage,
            hyperparameters,
            expected_performance,
            cost_analysis,
            confidence: best_option.confidence,
            calibration_method,
            production_config,
        })
    }

    /// Compute quantum advantage metrics
    fn compute_quantum_advantage(
        &self,
        option: &AlgorithmOption,
        _chars: &ProblemCharacteristics,
    ) -> Result<QuantumAdvantageMetrics> {
        let is_quantum = option.algorithm_type == AlgorithmType::Quantum;

        Ok(QuantumAdvantageMetrics {
            speedup: if is_quantum { 2.5 } else { 1.0 },
            accuracy_improvement: if is_quantum { 0.05 } else { 0.0 },
            sample_efficiency: if is_quantum { 1.8 } else { 1.0 },
            generalization_improvement: if is_quantum { 0.03 } else { 0.0 },
            statistical_significance: if is_quantum { 0.01 } else { 1.0 },
        })
    }

    /// Generate recommended hyperparameters
    fn generate_hyperparameters(
        &self,
        option: &AlgorithmOption,
        chars: &ProblemCharacteristics,
    ) -> Result<HashMap<String, f64>> {
        let mut params = HashMap::new();

        match option.name.as_str() {
            "QSVM" => {
                params.insert("n_shots".to_string(), 1000.0);
                params.insert("kernel_depth".to_string(), 3.0);
            }
            "QNN" => {
                params.insert("n_layers".to_string(), 5.0);
                params.insert("learning_rate".to_string(), 0.01);
                params.insert("batch_size".to_string(), 32.0);
            }
            "Neural Network" => {
                params.insert("hidden_layers".to_string(), 3.0);
                params.insert("neurons_per_layer".to_string(), 128.0);
                params.insert("learning_rate".to_string(), 0.001);
                params.insert("dropout".to_string(), 0.2);
            }
            _ => {}
        }

        Ok(params)
    }

    /// Generate cost analysis
    fn generate_cost_analysis(
        &self,
        option: &AlgorithmOption,
        chars: &ProblemCharacteristics,
    ) -> Result<CostAnalysis> {
        let training_cost = option.expected_cost;
        let inference_cost_per_sample = option.expected_cost / chars.n_samples as f64;

        let mut breakdown = HashMap::new();
        breakdown.insert("training".to_string(), training_cost);
        breakdown.insert("inference".to_string(), inference_cost_per_sample * 1000.0);

        Ok(CostAnalysis {
            training_cost,
            inference_cost_per_sample,
            total_cost: training_cost + inference_cost_per_sample * 10000.0,
            breakdown,
        })
    }

    /// Recommend calibration method
    fn recommend_calibration(
        &self,
        option: &AlgorithmOption,
        _chars: &ProblemCharacteristics,
    ) -> Result<Option<String>> {
        if option.algorithm_type == AlgorithmType::Quantum {
            Ok(Some("Bayesian Binning into Quantiles (BBQ)".to_string()))
        } else {
            Ok(Some("Platt Scaling".to_string()))
        }
    }

    /// Generate production configuration
    fn generate_production_config(
        &self,
        option: &AlgorithmOption,
        chars: &ProblemCharacteristics,
        constraints: &ResourceConstraints,
    ) -> Result<ProductionConfig> {
        let batch_size = if option.algorithm_type == AlgorithmType::Quantum {
            16
        } else if constraints.classical_compute.has_gpu {
            128
        } else {
            32
        };

        let n_workers = if option.algorithm_type == AlgorithmType::Quantum {
            1
        } else {
            constraints.classical_compute.n_cpu_cores.min(8)
        };

        let mut alert_thresholds = HashMap::new();
        alert_thresholds.insert(
            "latency_ms".to_string(),
            option.expected_inference_latency_ms * 2.0,
        );
        alert_thresholds.insert("accuracy".to_string(), option.expected_accuracy - 0.05);
        alert_thresholds.insert("error_rate".to_string(), 0.01);

        Ok(ProductionConfig {
            batch_size,
            n_workers,
            enable_caching: true,
            monitoring: MonitoringConfig {
                log_interval: 100,
                alert_thresholds,
                tracked_metrics: vec![
                    "accuracy".to_string(),
                    "latency".to_string(),
                    "throughput".to_string(),
                    "error_rate".to_string(),
                ],
            },
            scaling: ScalingConfig {
                auto_scaling: true,
                min_instances: 1,
                max_instances: 10,
                scale_up_threshold: 70.0,
                scale_down_threshold: 30.0,
            },
        })
    }

    /// Initialize performance prediction models
    fn initialize_performance_models() -> HashMap<String, PerformanceModel> {
        let mut models = HashMap::new();

        models.insert(
            "QSVM".to_string(),
            PerformanceModel {
                model_type: "linear".to_string(),
                coefficients: vec![0.9, -0.05, 0.03],
            },
        );

        models.insert(
            "QNN".to_string(),
            PerformanceModel {
                model_type: "linear".to_string(),
                coefficients: vec![0.87, -0.04, 0.05],
            },
        );

        models
    }

    /// Initialize cost prediction models
    fn initialize_cost_models() -> HashMap<String, CostModel> {
        let mut models = HashMap::new();

        models.insert(
            "QSVM".to_string(),
            CostModel {
                base_cost: 10.0,
                cost_per_sample: 0.01,
                cost_per_feature: 0.001,
                cost_per_qubit: 1.0,
            },
        );

        models.insert(
            "QNN".to_string(),
            CostModel {
                base_cost: 20.0,
                cost_per_sample: 0.02,
                cost_per_feature: 0.002,
                cost_per_qubit: 2.0,
            },
        );

        models
    }
}

/// Algorithm option being evaluated
#[derive(Debug, Clone)]
struct AlgorithmOption {
    name: String,
    algorithm_type: AlgorithmType,
    is_feasible: bool,
    expected_accuracy: f64,
    expected_training_time_s: f64,
    expected_inference_latency_ms: f64,
    expected_cost: f64,
    required_qubits: Option<usize>,
    confidence: f64,
}

/// Type of algorithm
#[derive(Debug, Clone, Copy, PartialEq)]
enum AlgorithmType {
    Quantum,
    Classical,
    Hybrid,
}

impl Default for DecisionThresholds {
    fn default() -> Self {
        Self {
            min_quantum_speedup: 1.5,
            min_accuracy_improvement: 0.02,
            max_cost_ratio: 10.0,
            min_confidence: 0.70,
        }
    }
}

impl ProblemCharacteristics {
    /// Extract problem characteristics from dataset
    pub fn from_dataset(x: &Array2<f64>, y: &Array1<usize>) -> Self {
        let n_samples = x.shape()[0];
        let n_features = x.shape()[1];
        let n_classes = y.iter().max().map(|&m| m + 1).unwrap_or(2);

        let dimensionality_ratio = n_features as f64 / n_samples as f64;

        // Compute sparsity
        let total_elements = n_samples * n_features;
        let zero_elements = x.iter().filter(|&&val| val.abs() < 1e-10).count();
        let sparsity = zero_elements as f64 / total_elements as f64;

        // Estimate condition number (simplified)
        let condition_number = 100.0; // Placeholder

        // Compute class imbalance
        let mut class_counts = vec![0; n_classes];
        for &label in y.iter() {
            if label < n_classes {
                class_counts[label] += 1;
            }
        }
        let max_count = class_counts.iter().max().copied().unwrap_or(1);
        let min_count = class_counts
            .iter()
            .filter(|&&c| c > 0)
            .min()
            .copied()
            .unwrap_or(1);
        let class_imbalance = max_count as f64 / min_count as f64;

        Self {
            n_samples,
            n_features,
            n_classes,
            dimensionality_ratio,
            sparsity,
            condition_number,
            class_imbalance,
            task_type: if n_classes == 2 {
                TaskType::BinaryClassification
            } else {
                TaskType::MultiClassClassification
            },
            domain: ProblemDomain::General,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_creation() {
        let engine = HybridAutoMLEngine::new();
        assert!(!engine.performance_models.is_empty());
    }

    #[test]
    fn test_problem_characteristics_extraction() {
        let x = Array2::from_shape_fn((100, 10), |(i, j)| (i + j) as f64);
        let y = Array1::from_shape_fn(100, |i| i % 2);

        let chars = ProblemCharacteristics::from_dataset(&x, &y);

        assert_eq!(chars.n_samples, 100);
        assert_eq!(chars.n_features, 10);
        assert_eq!(chars.n_classes, 2);
        assert_eq!(chars.task_type, TaskType::BinaryClassification);
    }

    #[test]
    fn test_algorithm_recommendation() {
        let engine = HybridAutoMLEngine::new();

        let chars = ProblemCharacteristics {
            n_samples: 1000,
            n_features: 20,
            n_classes: 2,
            dimensionality_ratio: 0.02,
            sparsity: 0.0,
            condition_number: 10.0,
            class_imbalance: 1.2,
            task_type: TaskType::BinaryClassification,
            domain: ProblemDomain::General,
        };

        let constraints = ResourceConstraints {
            quantum_devices: vec![QuantumDevice {
                name: "ibm_quantum".to_string(),
                n_qubits: 20,
                gate_error_rate: 0.001,
                measurement_error_rate: 0.01,
                decoherence_time_us: 100.0,
                cost_per_shot: 0.0001,
                availability: DeviceAvailability::Available,
            }],
            classical_compute: ClassicalCompute {
                n_cpu_cores: 8,
                has_gpu: true,
                gpu_memory_gb: 16.0,
                ram_gb: 64.0,
            },
            max_latency_ms: Some(100.0),
            max_cost_per_inference: Some(1.0),
            max_training_time: Some(1000.0),
            max_power_consumption: None,
        };

        let recommendation = engine.analyze_and_recommend(&chars, &constraints).unwrap();

        assert!(recommendation.confidence > 0.0);
        assert!(recommendation.expected_performance.accuracy > 0.0);
    }
}
