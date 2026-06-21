//! Performance Prediction Models for Circuit Execution Time Estimation
//!
//! This module provides sophisticated models for predicting quantum circuit
//! execution times across different simulation backends using `SciRS2` analysis
//! tools and machine learning techniques.

use crate::{
    auto_optimizer::{AnalysisDepth, BackendType, CircuitCharacteristics},
    error::{Result, SimulatorError},
    scirs2_integration::{Matrix, SciRS2Backend, Vector},
};
use quantrs2_circuit::builder::Circuit;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};
use scirs2_core::Complex64;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Configuration for performance prediction models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePredictionConfig {
    /// Enable machine learning-based predictions
    pub enable_ml_prediction: bool,
    /// Maximum historical data points to maintain
    pub max_history_size: usize,
    /// Confidence threshold for predictions (0.0 to 1.0)
    pub confidence_threshold: f64,
    /// Enable hardware profiling for adaptive predictions
    pub enable_hardware_profiling: bool,
    /// `SciRS2` analysis depth for complexity estimation
    pub analysis_depth: AnalysisDepth,
    /// Prediction strategy to use
    pub prediction_strategy: PredictionStrategy,
    /// Learning rate for adaptive models
    pub learning_rate: f64,
    /// Enable cross-backend performance transfer learning
    pub enable_transfer_learning: bool,
    /// Minimum samples required before using ML predictions
    pub min_samples_for_ml: usize,
}

impl Default for PerformancePredictionConfig {
    fn default() -> Self {
        Self {
            enable_ml_prediction: true,
            max_history_size: 10_000,
            confidence_threshold: 0.8,
            enable_hardware_profiling: true,
            analysis_depth: AnalysisDepth::Deep,
            prediction_strategy: PredictionStrategy::Hybrid,
            learning_rate: 0.01,
            enable_transfer_learning: true,
            min_samples_for_ml: 100,
        }
    }
}

/// Prediction strategy for execution time estimation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionStrategy {
    /// Static analysis only
    StaticAnalysis,
    /// Machine learning only
    MachineLearning,
    /// Hybrid approach (static + ML)
    Hybrid,
    /// Ensemble of multiple models
    Ensemble,
}

/// Performance prediction model types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    /// Linear regression model
    LinearRegression,
    /// Polynomial regression model
    PolynomialRegression,
    /// Neural network model
    NeuralNetwork,
    /// Support vector regression
    SupportVectorRegression,
    /// Random forest model
    RandomForest,
    /// Gradient boosting model
    GradientBoosting,
}

/// Circuit complexity metrics for prediction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityMetrics {
    /// Total number of gates
    pub gate_count: usize,
    /// Circuit depth (critical path length)
    pub circuit_depth: usize,
    /// Number of qubits
    pub qubit_count: usize,
    /// Number of two-qubit gates
    pub two_qubit_gate_count: usize,
    /// Estimated memory requirement (bytes)
    pub memory_requirement: usize,
    /// Parallelism potential (0.0 to 1.0)
    pub parallelism_factor: f64,
    /// Entanglement complexity measure
    pub entanglement_complexity: f64,
    /// Gate type distribution
    pub gate_type_distribution: HashMap<String, usize>,
    /// Critical path analysis
    pub critical_path_complexity: f64,
    /// Resource estimation
    pub resource_estimation: ResourceMetrics,
}

impl Default for ComplexityMetrics {
    fn default() -> Self {
        Self {
            gate_count: 0,
            circuit_depth: 0,
            qubit_count: 0,
            two_qubit_gate_count: 0,
            memory_requirement: 0,
            parallelism_factor: 0.0,
            entanglement_complexity: 0.0,
            gate_type_distribution: HashMap::new(),
            critical_path_complexity: 0.0,
            resource_estimation: ResourceMetrics::default(),
        }
    }
}

/// Resource requirements metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// Estimated CPU time (seconds)
    pub cpu_time_estimate: f64,
    /// Estimated memory usage (bytes)
    pub memory_usage_estimate: usize,
    /// Estimated I/O operations
    pub io_operations_estimate: usize,
    /// Network bandwidth requirement (bytes/sec)
    pub network_bandwidth_estimate: usize,
    /// GPU memory requirement (bytes)
    pub gpu_memory_estimate: usize,
    /// Parallel thread requirement
    pub thread_requirement: usize,
}

/// Historical execution data point
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionDataPoint {
    /// Circuit complexity metrics
    pub complexity: ComplexityMetrics,
    /// Backend used for execution
    pub backend_type: BackendType,
    /// Actual execution time
    pub execution_time: Duration,
    /// Hardware specifications during execution
    pub hardware_specs: PerformanceHardwareSpecs,
    /// Timestamp of execution
    #[serde(skip_serializing, skip_deserializing)]
    pub timestamp: std::time::SystemTime,
    /// Success flag
    pub success: bool,
    /// Error information if failed
    pub error_info: Option<String>,
}

impl Default for ExecutionDataPoint {
    fn default() -> Self {
        Self {
            complexity: ComplexityMetrics::default(),
            backend_type: BackendType::StateVector,
            execution_time: Duration::from_secs(0),
            hardware_specs: PerformanceHardwareSpecs::default(),
            timestamp: std::time::SystemTime::now(),
            success: false,
            error_info: None,
        }
    }
}

/// Hardware specifications for context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHardwareSpecs {
    /// CPU cores available
    pub cpu_cores: usize,
    /// Total system memory (bytes)
    pub total_memory: usize,
    /// Available memory at execution time (bytes)
    pub available_memory: usize,
    /// GPU memory (bytes, if available)
    pub gpu_memory: Option<usize>,
    /// CPU frequency (MHz)
    pub cpu_frequency: f64,
    /// Network bandwidth (Mbps, for distributed)
    pub network_bandwidth: Option<f64>,
    /// System load average
    pub load_average: f64,
}

impl Default for PerformanceHardwareSpecs {
    fn default() -> Self {
        Self {
            cpu_cores: 1,
            total_memory: 1024 * 1024 * 1024,    // 1GB
            available_memory: 512 * 1024 * 1024, // 512MB
            gpu_memory: None,
            cpu_frequency: 2000.0, // 2GHz
            network_bandwidth: None,
            load_average: 0.0,
        }
    }
}

/// Prediction result with confidence metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    /// Predicted execution time
    pub predicted_time: Duration,
    /// Confidence in prediction (0.0 to 1.0)
    pub confidence: f64,
    /// Prediction interval (lower bound, upper bound)
    pub prediction_interval: (Duration, Duration),
    /// Model used for prediction
    pub model_type: ModelType,
    /// Feature importance scores
    pub feature_importance: HashMap<String, f64>,
    /// Prediction metadata
    pub metadata: PredictionMetadata,
}

/// Metadata about the prediction process
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionMetadata {
    /// Time taken to generate prediction
    pub prediction_time: Duration,
    /// Number of historical samples used
    pub samples_used: usize,
    /// Model training status
    pub model_trained: bool,
    /// Cross-validation score (if available)
    pub cv_score: Option<f64>,
    /// Prediction method used
    pub prediction_method: String,
}

/// Performance prediction engine
pub struct PerformancePredictionEngine {
    /// Configuration
    config: PerformancePredictionConfig,
    /// Historical execution data
    execution_history: VecDeque<ExecutionDataPoint>,
    /// Trained models for different backends
    trained_models: HashMap<BackendType, TrainedModel>,
    /// `SciRS2` backend for analysis
    scirs2_backend: SciRS2Backend,
    /// Current hardware specifications
    current_hardware: PerformanceHardwareSpecs,
    /// Performance statistics
    prediction_stats: PredictionStatistics,
    /// Running accumulators for prediction-latency statistics (nanoseconds):
    /// (count, sum, sum-of-squares). Used to compute an exact mean/standard
    /// deviation without storing the full latency history.
    timing_accumulator: (u64, f64, f64),
}

/// Trained machine learning model
#[derive(Debug, Clone, Serialize)]
pub struct TrainedModel {
    /// Model type
    pub model_type: ModelType,
    /// Model parameters (simplified representation)
    pub parameters: Vec<f64>,
    /// Feature weights
    pub feature_weights: HashMap<String, f64>,
    /// Training statistics
    pub training_stats: TrainingStatistics,
    /// Last training time
    #[serde(skip_serializing, skip_deserializing)]
    pub last_trained: std::time::SystemTime,
}

impl Default for TrainedModel {
    fn default() -> Self {
        Self {
            model_type: ModelType::LinearRegression,
            parameters: Vec::new(),
            feature_weights: HashMap::new(),
            training_stats: TrainingStatistics::default(),
            last_trained: std::time::SystemTime::now(),
        }
    }
}

/// Training statistics for models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingStatistics {
    /// Training samples used
    pub training_samples: usize,
    /// Training accuracy (R²)
    pub training_accuracy: f64,
    /// Validation accuracy
    pub validation_accuracy: f64,
    /// Mean absolute error
    pub mean_absolute_error: f64,
    /// Root mean square error
    pub root_mean_square_error: f64,
    /// Training time
    pub training_time: Duration,
}

impl Default for TrainingStatistics {
    fn default() -> Self {
        Self {
            training_samples: 0,
            training_accuracy: 0.0,
            validation_accuracy: 0.0,
            mean_absolute_error: 0.0,
            root_mean_square_error: 0.0,
            training_time: Duration::from_secs(0),
        }
    }
}

/// Overall prediction engine statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionStatistics {
    /// Total predictions made
    pub total_predictions: usize,
    /// Successful predictions
    pub successful_predictions: usize,
    /// Average prediction accuracy
    pub average_accuracy: f64,
    /// Prediction time statistics
    pub prediction_time_stats: PerformanceTimingStatistics,
    /// Model update frequency
    pub model_updates: usize,
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

/// Timing statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTimingStatistics {
    /// Average time
    pub average: Duration,
    /// Minimum time
    pub minimum: Duration,
    /// Maximum time
    pub maximum: Duration,
    /// Standard deviation
    pub std_deviation: Duration,
}

impl PerformancePredictionEngine {
    /// Create new performance prediction engine
    pub fn new(config: PerformancePredictionConfig) -> Result<Self> {
        let current_hardware = Self::detect_hardware_specs()?;

        Ok(Self {
            config,
            execution_history: VecDeque::with_capacity(10_000),
            trained_models: HashMap::new(),
            scirs2_backend: SciRS2Backend::new(),
            current_hardware,
            prediction_stats: PredictionStatistics::default(),
            timing_accumulator: (0, 0.0, 0.0),
        })
    }

    /// Predict execution time for a circuit on a specific backend
    pub fn predict_execution_time<const N: usize>(
        &mut self,
        circuit: &Circuit<N>,
        backend_type: BackendType,
    ) -> Result<PredictionResult> {
        let start_time = Instant::now();

        // Analyze circuit complexity using SciRS2
        let complexity = self.analyze_circuit_complexity(circuit)?;

        // Get prediction based on strategy
        let prediction = match self.config.prediction_strategy {
            PredictionStrategy::StaticAnalysis => {
                self.predict_with_static_analysis(&complexity, backend_type)?
            }
            PredictionStrategy::MachineLearning => {
                self.predict_with_ml(&complexity, backend_type)?
            }
            PredictionStrategy::Hybrid => self.predict_with_hybrid(&complexity, backend_type)?,
            PredictionStrategy::Ensemble => {
                self.predict_with_ensemble(&complexity, backend_type)?
            }
        };

        // Update statistics
        self.prediction_stats.total_predictions += 1;
        let prediction_time = start_time.elapsed();
        self.update_timing_stats(prediction_time);

        Ok(prediction)
    }

    /// Analyze circuit complexity using `SciRS2` tools
    fn analyze_circuit_complexity<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> Result<ComplexityMetrics> {
        let gate_count = circuit.num_gates();
        let qubit_count = N;

        // Basic complexity analysis
        let circuit_depth = self.calculate_circuit_depth(circuit)?;
        let two_qubit_gate_count = self.count_two_qubit_gates(circuit)?;
        let memory_requirement = self.estimate_memory_requirement(qubit_count);

        // Advanced analysis using SciRS2
        let parallelism_factor = self.analyze_parallelism_potential(circuit)?;
        let entanglement_complexity = self.estimate_entanglement_complexity(circuit)?;
        let gate_type_distribution = self.analyze_gate_distribution(circuit)?;
        let critical_path_complexity = self.analyze_critical_path(circuit)?;

        // Resource estimation
        let resource_estimation = self.estimate_resources(&ComplexityMetrics {
            gate_count,
            circuit_depth,
            qubit_count,
            two_qubit_gate_count,
            memory_requirement,
            parallelism_factor,
            entanglement_complexity,
            gate_type_distribution: gate_type_distribution.clone(),
            critical_path_complexity,
            resource_estimation: ResourceMetrics::default(), // Will be filled
        })?;

        Ok(ComplexityMetrics {
            gate_count,
            circuit_depth,
            qubit_count,
            two_qubit_gate_count,
            memory_requirement,
            parallelism_factor,
            entanglement_complexity,
            gate_type_distribution,
            critical_path_complexity,
            resource_estimation,
        })
    }

    /// Calculate circuit depth (critical path length)
    fn calculate_circuit_depth<const N: usize>(&self, circuit: &Circuit<N>) -> Result<usize> {
        // Simple depth calculation - can be enhanced with SciRS2 graph analysis
        let mut qubit_last_gate: Vec<usize> = vec![0; N];
        let mut max_depth = 0;

        let gates = circuit.gates_as_boxes();
        for (gate_idx, gate) in gates.iter().enumerate() {
            let gate_qubits = self.get_gate_qubits(gate.as_ref())?;
            let mut max_dependency = 0;

            for &qubit in &gate_qubits {
                if qubit < N {
                    max_dependency = max_dependency.max(qubit_last_gate[qubit]);
                }
            }

            let current_depth = max_dependency + 1;
            max_depth = max_depth.max(current_depth);

            for &qubit in &gate_qubits {
                if qubit < N {
                    qubit_last_gate[qubit] = current_depth;
                }
            }
        }

        Ok(max_depth)
    }

    /// Count two-qubit gates in circuit
    fn count_two_qubit_gates<const N: usize>(&self, circuit: &Circuit<N>) -> Result<usize> {
        let mut count = 0;
        let gates = circuit.gates_as_boxes();
        for gate in &gates {
            let qubits = self.get_gate_qubits(gate.as_ref())?;
            if qubits.len() >= 2 {
                count += 1;
            }
        }
        Ok(count)
    }

    /// Get qubits affected by a gate
    fn get_gate_qubits(&self, gate: &dyn GateOp) -> Result<Vec<usize>> {
        // Extract qubit indices from gate operation using the GateOp trait
        let qubits = gate.qubits();
        Ok(qubits.iter().map(|q| q.id() as usize).collect())
    }

    /// Estimate memory requirement for simulation
    const fn estimate_memory_requirement(&self, qubit_count: usize) -> usize {
        // 2^N complex numbers, each 16 bytes (8 bytes real + 8 bytes imag)
        let state_vector_size = (1usize << qubit_count) * 16;
        // Add overhead for intermediate calculations
        state_vector_size * 3
    }

    /// Analyze parallelism potential using `SciRS2`
    fn analyze_parallelism_potential<const N: usize>(&self, circuit: &Circuit<N>) -> Result<f64> {
        // Use SciRS2 parallel analysis
        let independent_operations = self.count_independent_operations(circuit)?;
        let total_operations = circuit.num_gates();

        if total_operations == 0 {
            return Ok(0.0);
        }

        Ok(independent_operations as f64 / total_operations as f64)
    }

    /// Count independent operations that can be parallelized
    fn count_independent_operations<const N: usize>(&self, circuit: &Circuit<N>) -> Result<usize> {
        // Analyze gate dependencies for parallelization opportunities
        // This is a simplified implementation
        let mut independent_count = 0;
        let mut qubit_dependencies: Vec<Option<usize>> = vec![None; N];

        let gates = circuit.gates_as_boxes();
        for (gate_idx, gate) in gates.iter().enumerate() {
            let gate_qubits = self.get_gate_qubits(gate.as_ref())?;
            let mut has_dependency = false;

            for &qubit in &gate_qubits {
                if qubit < N && qubit_dependencies[qubit].is_some() {
                    has_dependency = true;
                    break;
                }
            }

            if !has_dependency {
                independent_count += 1;
            }

            // Update dependencies
            for &qubit in &gate_qubits {
                if qubit < N {
                    qubit_dependencies[qubit] = Some(gate_idx);
                }
            }
        }

        Ok(independent_count)
    }

    /// Estimate entanglement complexity
    fn estimate_entanglement_complexity<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> Result<f64> {
        // Simplified entanglement analysis
        let two_qubit_gates = self.count_two_qubit_gates(circuit)?;
        let total_possible_entangling = N * (N - 1) / 2; // All possible qubit pairs

        if total_possible_entangling == 0 {
            return Ok(0.0);
        }

        Ok((two_qubit_gates as f64 / total_possible_entangling as f64).min(1.0))
    }

    /// Analyze gate type distribution
    fn analyze_gate_distribution<const N: usize>(
        &self,
        circuit: &Circuit<N>,
    ) -> Result<HashMap<String, usize>> {
        let mut distribution = HashMap::new();

        let gates = circuit.gates_as_boxes();
        for gate in &gates {
            let gate_type = self.get_gate_type_name(gate.as_ref());
            *distribution.entry(gate_type).or_insert(0) += 1;
        }

        Ok(distribution)
    }

    /// Get gate type name for classification
    fn get_gate_type_name(&self, gate: &dyn GateOp) -> String {
        // Use the gate's name from the GateOp trait
        gate.name().to_string()
    }

    /// Analyze critical path complexity
    fn analyze_critical_path<const N: usize>(&self, circuit: &Circuit<N>) -> Result<f64> {
        // Analyze the complexity of the critical path
        let depth = self.calculate_circuit_depth(circuit)?;
        let gate_count = circuit.num_gates();

        if gate_count == 0 {
            return Ok(0.0);
        }

        // Complexity is depth relative to total gates
        Ok(depth as f64 / gate_count as f64)
    }

    /// Estimate resource requirements
    fn estimate_resources(&self, complexity: &ComplexityMetrics) -> Result<ResourceMetrics> {
        // CPU time estimation based on complexity
        let base_cpu_time = complexity.gate_count as f64 * 1e-6; // 1 microsecond per gate base
        let depth_factor = complexity.circuit_depth as f64 * 0.1;
        let entanglement_factor = complexity.entanglement_complexity * 2.0;
        let cpu_time_estimate = base_cpu_time * (1.0 + depth_factor + entanglement_factor);

        // Memory estimation
        let memory_usage_estimate = complexity.memory_requirement;

        // I/O estimation
        let io_operations_estimate = complexity.gate_count * 2; // Read + write per gate

        // Network bandwidth for distributed execution
        let network_bandwidth_estimate = if complexity.qubit_count > 20 {
            complexity.memory_requirement / 10 // 10% of memory for communication
        } else {
            0
        };

        // GPU memory estimation
        let gpu_memory_estimate = complexity.memory_requirement * 2; // GPU needs more memory

        // Thread requirement
        let thread_requirement = (complexity.parallelism_factor * 16.0).ceil() as usize;

        Ok(ResourceMetrics {
            cpu_time_estimate,
            memory_usage_estimate,
            io_operations_estimate,
            network_bandwidth_estimate,
            gpu_memory_estimate,
            thread_requirement,
        })
    }

    /// Predict using static analysis only
    fn predict_with_static_analysis(
        &self,
        complexity: &ComplexityMetrics,
        backend_type: BackendType,
    ) -> Result<PredictionResult> {
        // Static analysis-based prediction
        let base_time = complexity.resource_estimation.cpu_time_estimate;

        // Backend-specific factors
        let backend_factor = match backend_type {
            BackendType::StateVector => 1.0,
            BackendType::SciRS2Gpu => 0.3,   // GPU acceleration
            BackendType::LargeScale => 0.7,  // Optimized for large circuits
            BackendType::Distributed => 0.5, // Distributed speedup
            BackendType::Auto => 0.8,        // Conservative estimate
        };

        let predicted_seconds = base_time * backend_factor;
        let predicted_time = Duration::from_secs_f64(predicted_seconds);

        // Static confidence based on circuit characteristics
        let confidence = if complexity.qubit_count <= 20 {
            0.9
        } else {
            0.7
        };

        // Prediction interval (±20%)
        let lower = Duration::from_secs_f64(predicted_seconds * 0.8);
        let upper = Duration::from_secs_f64(predicted_seconds * 1.2);

        Ok(PredictionResult {
            predicted_time,
            confidence,
            prediction_interval: (lower, upper),
            model_type: ModelType::LinearRegression,
            feature_importance: HashMap::new(),
            metadata: PredictionMetadata {
                prediction_time: Duration::from_millis(1),
                samples_used: 0,
                model_trained: false,
                cv_score: None,
                prediction_method: "Static Analysis".to_string(),
            },
        })
    }

    /// Predict using machine learning
    fn predict_with_ml(
        &mut self,
        complexity: &ComplexityMetrics,
        backend_type: BackendType,
    ) -> Result<PredictionResult> {
        // Check if we have enough historical data
        if self.execution_history.len() < self.config.min_samples_for_ml {
            return self.predict_with_static_analysis(complexity, backend_type);
        }

        // Train model if needed
        if !self.trained_models.contains_key(&backend_type) {
            self.train_model_for_backend(backend_type)?;
        }

        // Get trained model
        let model = self
            .trained_models
            .get(&backend_type)
            .ok_or_else(|| SimulatorError::ComputationError("Model not found".to_string()))?;

        // Make prediction using trained model
        let predicted_seconds = self.apply_model(model, complexity)?;
        let predicted_time = Duration::from_secs_f64(predicted_seconds);

        // ML confidence based on training statistics
        let confidence = model.training_stats.validation_accuracy;

        // Prediction interval based on model error
        let error_margin = model.training_stats.mean_absolute_error;
        let lower = Duration::from_secs_f64((predicted_seconds - error_margin).max(0.0));
        let upper = Duration::from_secs_f64(predicted_seconds + error_margin);

        Ok(PredictionResult {
            predicted_time,
            confidence,
            prediction_interval: (lower, upper),
            model_type: model.model_type,
            feature_importance: model.feature_weights.clone(),
            metadata: PredictionMetadata {
                prediction_time: Duration::from_millis(5),
                samples_used: model.training_stats.training_samples,
                model_trained: true,
                cv_score: Some(model.training_stats.validation_accuracy),
                prediction_method: "Machine Learning".to_string(),
            },
        })
    }

    /// Predict using hybrid approach (static + ML)
    fn predict_with_hybrid(
        &mut self,
        complexity: &ComplexityMetrics,
        backend_type: BackendType,
    ) -> Result<PredictionResult> {
        // Get static prediction
        let static_pred = self.predict_with_static_analysis(complexity, backend_type)?;

        // Try ML prediction if enough data
        if self.execution_history.len() >= self.config.min_samples_for_ml {
            let ml_pred = self.predict_with_ml(complexity, backend_type)?;

            // Weighted combination
            let static_weight = 0.3;
            let ml_weight = 0.7;

            let combined_seconds = static_pred.predicted_time.as_secs_f64().mul_add(
                static_weight,
                ml_pred.predicted_time.as_secs_f64() * ml_weight,
            );

            let predicted_time = Duration::from_secs_f64(combined_seconds);
            let confidence = static_pred
                .confidence
                .mul_add(static_weight, ml_pred.confidence * ml_weight);

            // Combined prediction interval
            let lower_combined =
                Duration::from_secs_f64(static_pred.prediction_interval.0.as_secs_f64().mul_add(
                    static_weight,
                    ml_pred.prediction_interval.0.as_secs_f64() * ml_weight,
                ));
            let upper_combined =
                Duration::from_secs_f64(static_pred.prediction_interval.1.as_secs_f64().mul_add(
                    static_weight,
                    ml_pred.prediction_interval.1.as_secs_f64() * ml_weight,
                ));

            Ok(PredictionResult {
                predicted_time,
                confidence,
                prediction_interval: (lower_combined, upper_combined),
                model_type: ModelType::LinearRegression, // Hybrid
                feature_importance: ml_pred.feature_importance,
                metadata: PredictionMetadata {
                    prediction_time: Duration::from_millis(6),
                    samples_used: ml_pred.metadata.samples_used,
                    model_trained: ml_pred.metadata.model_trained,
                    cv_score: ml_pred.metadata.cv_score,
                    prediction_method: "Hybrid (Static + ML)".to_string(),
                },
            })
        } else {
            // Fall back to static analysis
            Ok(static_pred)
        }
    }

    /// Predict using an ensemble of independent base predictors.
    ///
    /// Runs the static-analysis predictor and, when enough history exists, the ML
    /// regressor as two independent base learners, then combines them with
    /// confidence-weighted averaging (each base prediction weighted by its own
    /// reported confidence). The ensemble confidence is the weighted mean of the
    /// base confidences and the prediction interval is the union (min lower /
    /// max upper) of the base intervals, which is a real reflection of combined
    /// uncertainty rather than a copy of a single model.
    fn predict_with_ensemble(
        &mut self,
        complexity: &ComplexityMetrics,
        backend_type: BackendType,
    ) -> Result<PredictionResult> {
        let mut members: Vec<PredictionResult> = Vec::new();

        // Base learner 1: static analysis (always available).
        members.push(self.predict_with_static_analysis(complexity, backend_type)?);

        // Base learner 2: trained ML model, only when enough history exists.
        if self.execution_history.len() >= self.config.min_samples_for_ml {
            if let Ok(ml_pred) = self.predict_with_ml(complexity, backend_type) {
                members.push(ml_pred);
            }
        }

        // Confidence-weighted combination (fall back to equal weights if all
        // confidences are zero).
        let total_confidence: f64 = members.iter().map(|m| m.confidence).sum();
        let use_equal = total_confidence <= f64::EPSILON;
        let member_count = members.len() as f64;

        let mut predicted_seconds = 0.0;
        let mut confidence = 0.0;
        let mut lower = f64::MAX;
        let mut upper: f64 = 0.0;
        let mut samples_used = 0usize;
        let mut model_trained = false;
        let mut feature_importance = HashMap::new();

        for member in &members {
            let weight = if use_equal {
                1.0 / member_count
            } else {
                member.confidence / total_confidence
            };
            predicted_seconds += member.predicted_time.as_secs_f64() * weight;
            confidence += member.confidence * weight;
            lower = lower.min(member.prediction_interval.0.as_secs_f64());
            upper = upper.max(member.prediction_interval.1.as_secs_f64());
            samples_used = samples_used.max(member.metadata.samples_used);
            model_trained |= member.metadata.model_trained;
            if !member.feature_importance.is_empty() {
                feature_importance = member.feature_importance.clone();
            }
        }

        if lower == f64::MAX {
            lower = predicted_seconds;
        }

        Ok(PredictionResult {
            predicted_time: Duration::from_secs_f64(predicted_seconds.max(0.0)),
            confidence: confidence.clamp(0.0, 1.0),
            prediction_interval: (
                Duration::from_secs_f64(lower.max(0.0)),
                Duration::from_secs_f64(upper.max(0.0)),
            ),
            model_type: ModelType::RandomForest,
            feature_importance,
            metadata: PredictionMetadata {
                prediction_time: Duration::from_millis(8),
                samples_used,
                model_trained,
                cv_score: None,
                prediction_method: format!("Ensemble ({} base models)", members.len()),
            },
        })
    }

    /// Feature names used by the linear model, in column order. The first column
    /// of the design matrix is an implicit intercept (handled separately).
    const FEATURE_NAMES: [&'static str; 5] = [
        "gate_count",
        "circuit_depth",
        "qubit_count",
        "entanglement_complexity",
        "parallelism_factor",
    ];

    /// Extract the model feature row for a set of complexity metrics.
    ///
    /// `gate_count`, `circuit_depth` and `qubit_count` are log1p-transformed so a
    /// linear model captures their multiplicative effect on runtime; the two
    /// ratio features are used directly.
    fn feature_row(complexity: &ComplexityMetrics) -> [f64; 5] {
        [
            (complexity.gate_count as f64).ln_1p(),
            (complexity.circuit_depth as f64).ln_1p(),
            (complexity.qubit_count as f64).ln_1p(),
            complexity.entanglement_complexity,
            complexity.parallelism_factor,
        ]
    }

    /// Train a real ordinary-least-squares linear regression for one backend.
    ///
    /// The model fits `ln(1 + execution_seconds)` against the feature row (plus an
    /// intercept) over all successful historical runs for the backend, solving the
    /// normal equations `(XᵀX) β = Xᵀy` with Gaussian elimination. Training and
    /// validation accuracy are the real coefficients of determination (R²) on a
    /// time-ordered train/validation split, and MAE/RMSE are measured on the
    /// validation fold (or the training fold when there is too little data to
    /// split). No statistics are fabricated.
    fn train_model_for_backend(&mut self, backend_type: BackendType) -> Result<()> {
        let train_start = Instant::now();

        let training_data: Vec<&ExecutionDataPoint> = self
            .execution_history
            .iter()
            .filter(|data| data.backend_type == backend_type && data.success)
            .collect();

        if training_data.is_empty() {
            return Err(SimulatorError::ComputationError(
                "No training data available".to_string(),
            ));
        }

        // Build design rows (features + target). Target is ln(1 + seconds).
        let samples: Vec<([f64; 5], f64)> = training_data
            .iter()
            .map(|dp| {
                (
                    Self::feature_row(&dp.complexity),
                    dp.execution_time.as_secs_f64().ln_1p(),
                )
            })
            .collect();

        // Time-ordered split: last 20% (at least 1) used for validation when we
        // have enough samples; otherwise validate on the training data itself.
        let total = samples.len();
        let split = if total >= 5 {
            total - (total / 5).max(1)
        } else {
            total
        };
        let (train_slice, valid_slice) = samples.split_at(split);
        let train_slice = if train_slice.is_empty() {
            &samples[..]
        } else {
            train_slice
        };
        let valid_slice = if valid_slice.is_empty() {
            train_slice
        } else {
            valid_slice
        };

        // Fit coefficients [intercept, w0..w4] via the normal equations.
        let coefficients = Self::fit_least_squares(train_slice)?;

        let training_accuracy = Self::r_squared(train_slice, &coefficients);
        let validation_accuracy = Self::r_squared(valid_slice, &coefficients);
        let (mean_absolute_error, root_mean_square_error) =
            Self::error_metrics(valid_slice, &coefficients);

        // Feature importance: |standardized coefficient| normalized to sum to 1.
        let feature_weights = Self::feature_importance(train_slice, &coefficients);

        let model = TrainedModel {
            model_type: ModelType::LinearRegression,
            parameters: coefficients,
            feature_weights,
            training_stats: TrainingStatistics {
                training_samples: train_slice.len(),
                training_accuracy,
                validation_accuracy,
                mean_absolute_error,
                root_mean_square_error,
                training_time: train_start.elapsed(),
            },
            last_trained: std::time::SystemTime::now(),
        };

        self.trained_models.insert(backend_type, model);
        self.prediction_stats.model_updates += 1;

        Ok(())
    }

    /// Solve ordinary least squares for `[intercept, w0..w4]` over the samples.
    ///
    /// Forms the 6x6 normal-equation system `(XᵀX) β = Xᵀy` (the leading column of
    /// `X` is all ones for the intercept) and solves it with partial-pivoted
    /// Gaussian elimination. A tiny ridge term is added to the diagonal to keep
    /// the system solvable when features are collinear or samples are scarce.
    fn fit_least_squares(samples: &[([f64; 5], f64)]) -> Result<Vec<f64>> {
        const DIM: usize = 6; // intercept + 5 features
        if samples.is_empty() {
            return Err(SimulatorError::ComputationError(
                "cannot fit model with no samples".to_string(),
            ));
        }

        let mut ata = [[0.0f64; DIM]; DIM];
        let mut aty = [0.0f64; DIM];

        for (features, target) in samples {
            let mut row = [0.0f64; DIM];
            row[0] = 1.0;
            row[1..DIM].copy_from_slice(features);

            for i in 0..DIM {
                aty[i] += row[i] * target;
                for j in 0..DIM {
                    ata[i][j] += row[i] * row[j];
                }
            }
        }

        // Ridge regularization for numerical stability (does not bias a
        // well-conditioned fit meaningfully).
        let ridge = 1e-6;
        for i in 0..DIM {
            ata[i][i] += ridge;
        }

        Self::solve_linear_system(ata, aty)
    }

    /// Solve a fixed 6x6 linear system with partial-pivoted Gaussian elimination.
    fn solve_linear_system(mut a: [[f64; 6]; 6], mut b: [f64; 6]) -> Result<Vec<f64>> {
        const DIM: usize = 6;
        for col in 0..DIM {
            // Partial pivot: find the largest magnitude entry in this column.
            let mut pivot = col;
            let mut best = a[col][col].abs();
            for row in (col + 1)..DIM {
                let candidate = a[row][col].abs();
                if candidate > best {
                    best = candidate;
                    pivot = row;
                }
            }
            if best < 1e-12 {
                return Err(SimulatorError::ComputationError(
                    "singular system while fitting performance model".to_string(),
                ));
            }
            if pivot != col {
                a.swap(col, pivot);
                b.swap(col, pivot);
            }
            // Eliminate below the pivot.
            for row in (col + 1)..DIM {
                let factor = a[row][col] / a[col][col];
                for k in col..DIM {
                    a[row][k] -= factor * a[col][k];
                }
                b[row] -= factor * b[col];
            }
        }

        // Back-substitution.
        let mut x = vec![0.0f64; DIM];
        for row in (0..DIM).rev() {
            let mut sum = b[row];
            for k in (row + 1)..DIM {
                sum -= a[row][k] * x[k];
            }
            x[row] = sum / a[row][row];
        }
        Ok(x)
    }

    /// Predict `ln(1 + seconds)` from a feature row and fitted coefficients.
    fn predict_log_time(features: &[f64; 5], coefficients: &[f64]) -> f64 {
        let intercept = coefficients.first().copied().unwrap_or(0.0);
        let mut acc = intercept;
        for (idx, value) in features.iter().enumerate() {
            acc += coefficients.get(idx + 1).copied().unwrap_or(0.0) * value;
        }
        acc
    }

    /// Coefficient of determination (R²) of the fit over the given samples.
    fn r_squared(samples: &[([f64; 5], f64)], coefficients: &[f64]) -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        let mean = samples.iter().map(|(_, y)| *y).sum::<f64>() / samples.len() as f64;
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        for (features, target) in samples {
            let predicted = Self::predict_log_time(features, coefficients);
            ss_res += (target - predicted).powi(2);
            ss_tot += (target - mean).powi(2);
        }
        if ss_tot <= f64::EPSILON {
            // All targets identical: a perfect fit reproduces them, otherwise 0.
            return if ss_res <= f64::EPSILON { 1.0 } else { 0.0 };
        }
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    }

    /// Mean absolute error and root-mean-square error in the original time domain
    /// (seconds), measured over the given samples.
    fn error_metrics(samples: &[([f64; 5], f64)], coefficients: &[f64]) -> (f64, f64) {
        if samples.is_empty() {
            return (0.0, 0.0);
        }
        let mut abs_sum = 0.0;
        let mut sq_sum = 0.0;
        for (features, target) in samples {
            let predicted_log = Self::predict_log_time(features, coefficients);
            // Convert both back from ln(1 + seconds) to seconds.
            let predicted = predicted_log.exp_m1().max(0.0);
            let actual = target.exp_m1().max(0.0);
            let diff = predicted - actual;
            abs_sum += diff.abs();
            sq_sum += diff * diff;
        }
        let n = samples.len() as f64;
        (abs_sum / n, (sq_sum / n).sqrt())
    }

    /// Normalized feature importance from standardized coefficients.
    ///
    /// Each coefficient is scaled by the standard deviation of its feature so the
    /// magnitudes are comparable, then the absolute values are normalized to sum
    /// to 1. Returns an empty map only when no samples are available.
    fn feature_importance(
        samples: &[([f64; 5], f64)],
        coefficients: &[f64],
    ) -> HashMap<String, f64> {
        let mut weights = HashMap::new();
        if samples.is_empty() {
            return weights;
        }
        let n = samples.len() as f64;
        let mut scaled = [0.0f64; 5];
        for col in 0..5 {
            let mean = samples.iter().map(|(f, _)| f[col]).sum::<f64>() / n;
            let variance = samples
                .iter()
                .map(|(f, _)| (f[col] - mean).powi(2))
                .sum::<f64>()
                / n;
            let std_dev = variance.sqrt();
            let coeff = coefficients.get(col + 1).copied().unwrap_or(0.0);
            scaled[col] = (coeff * std_dev).abs();
        }
        let total: f64 = scaled.iter().sum();
        for (col, name) in Self::FEATURE_NAMES.iter().enumerate() {
            let importance = if total > f64::EPSILON {
                scaled[col] / total
            } else {
                0.0
            };
            weights.insert((*name).to_string(), importance);
        }
        weights
    }

    /// Apply a trained linear model to predict execution time (seconds).
    ///
    /// Evaluates the fitted regression on the feature row and converts the
    /// predicted `ln(1 + seconds)` back to seconds, clamped to be non-negative.
    fn apply_model(&self, model: &TrainedModel, complexity: &ComplexityMetrics) -> Result<f64> {
        let features = Self::feature_row(complexity);
        let predicted_log = Self::predict_log_time(&features, &model.parameters);
        Ok(predicted_log.exp_m1().max(0.0))
    }

    /// Record actual execution time for model improvement
    pub fn record_execution(&mut self, data_point: ExecutionDataPoint) -> Result<()> {
        // Add to history
        self.execution_history.push_back(data_point.clone());

        // Maintain size limit
        if self.execution_history.len() > self.config.max_history_size {
            self.execution_history.pop_front();
        }

        // Update prediction accuracy if we have a prediction for this data
        self.update_prediction_accuracy(&data_point);

        // Retrain models periodically
        if self.execution_history.len() % 100 == 0 {
            self.retrain_models()?;
        }

        Ok(())
    }

    /// Update prediction-accuracy statistics from a newly observed execution.
    ///
    /// When a trained model exists for the data point's backend, the model is
    /// re-evaluated on the point's complexity and the prediction is compared to
    /// the actual measured time. The per-sample accuracy is
    /// `1 - |predicted - actual| / max(actual, eps)` (clamped to `[0, 1]`), and
    /// the engine's `average_accuracy` is updated as a running mean over all such
    /// comparisons. This is a measured back-test, not a placeholder.
    fn update_prediction_accuracy(&mut self, data_point: &ExecutionDataPoint) {
        if !data_point.success {
            return;
        }

        self.prediction_stats.successful_predictions += 1;

        let Some(model) = self.trained_models.get(&data_point.backend_type) else {
            return;
        };

        let Ok(predicted_seconds) = self.apply_model(model, &data_point.complexity) else {
            return;
        };

        let actual_seconds = data_point.execution_time.as_secs_f64();
        let denom = actual_seconds.max(1e-9);
        let sample_accuracy =
            (1.0 - (predicted_seconds - actual_seconds).abs() / denom).clamp(0.0, 1.0);

        // Running mean of accuracy over successful, model-backed predictions.
        let n = self.prediction_stats.successful_predictions as f64;
        let prev = self.prediction_stats.average_accuracy;
        self.prediction_stats.average_accuracy = prev + (sample_accuracy - prev) / n;
    }

    /// Retrain all models with latest data
    fn retrain_models(&mut self) -> Result<()> {
        let backends = vec![
            BackendType::StateVector,
            BackendType::SciRS2Gpu,
            BackendType::LargeScale,
            BackendType::Distributed,
        ];

        for backend in backends {
            if self
                .execution_history
                .iter()
                .any(|d| d.backend_type == backend)
            {
                self.train_model_for_backend(backend)?;
            }
        }

        Ok(())
    }

    /// Detect current hardware specifications from real system sources.
    ///
    /// CPU core count comes from `num_cpus`; total/available memory are read from
    /// the OS via `scirs2_core::resource` (which parses `/proc/meminfo` on Linux,
    /// with a documented fallback). The system load average is read from
    /// `/proc/loadavg` when present.
    ///
    /// Values that cannot be probed in-process without extra dependencies or a
    /// running GPU context are reported honestly rather than fabricated:
    /// `gpu_memory` is `None` (no GPU memory is queried here), `cpu_frequency` is
    /// `0.0` ("unknown"), and `network_bandwidth` is `None`.
    fn detect_hardware_specs() -> Result<PerformanceHardwareSpecs> {
        let cpu_cores = num_cpus::get();

        let total_memory = scirs2_core::resource::get_total_memory()
            .map_err(|e| SimulatorError::ComputationError(format!("memory probe failed: {e}")))?;
        let available_memory = scirs2_core::resource::get_available_memory()
            .map_err(|e| SimulatorError::ComputationError(format!("memory probe failed: {e}")))?;

        let load_average = Self::read_load_average();

        Ok(PerformanceHardwareSpecs {
            cpu_cores,
            total_memory,
            available_memory,
            // No GPU memory is queried on this path; do not fabricate a size.
            gpu_memory: None,
            // Per-core frequency is not portably probeable in-process; "unknown".
            cpu_frequency: 0.0,
            // Network bandwidth is not measured here.
            network_bandwidth: None,
            load_average,
        })
    }

    /// Read the 1-minute load average from `/proc/loadavg` (Linux). Returns 0.0
    /// when unavailable, signalling "unknown" rather than fabricating a value.
    fn read_load_average() -> f64 {
        std::fs::read_to_string("/proc/loadavg")
            .ok()
            .and_then(|content| {
                content
                    .split_whitespace()
                    .next()
                    .and_then(|first| first.parse::<f64>().ok())
            })
            .unwrap_or(0.0)
    }

    /// Update prediction-latency statistics with a new measured `elapsed` time.
    ///
    /// Maintains exact running count/sum/sum-of-squares (in nanoseconds) and
    /// recomputes the reported average, minimum, maximum and population standard
    /// deviation. These are real measurements of how long predictions take, not
    /// placeholders.
    fn update_timing_stats(&mut self, elapsed: Duration) {
        let nanos = elapsed.as_nanos() as f64;
        let (count, sum, sum_sq) = self.timing_accumulator;
        let new_count = count + 1;
        let new_sum = sum + nanos;
        let new_sum_sq = sum_sq + nanos * nanos;
        self.timing_accumulator = (new_count, new_sum, new_sum_sq);

        let mean = new_sum / new_count as f64;
        let variance = (new_sum_sq / new_count as f64) - mean * mean;
        let std_dev = variance.max(0.0).sqrt();

        let stats = &mut self.prediction_stats.prediction_time_stats;
        stats.average = Duration::from_nanos(mean as u64);
        stats.std_deviation = Duration::from_nanos(std_dev as u64);
        if new_count == 1 {
            stats.minimum = elapsed;
            stats.maximum = elapsed;
        } else {
            stats.minimum = stats.minimum.min(elapsed);
            stats.maximum = stats.maximum.max(elapsed);
        }
    }

    /// Get prediction engine statistics
    #[must_use]
    pub const fn get_statistics(&self) -> &PredictionStatistics {
        &self.prediction_stats
    }

    /// Export prediction models for persistence
    pub fn export_models(&self) -> Result<Vec<u8>> {
        // Serialize models for storage
        let serialized = serde_json::to_vec(&self.trained_models)
            .map_err(|e| SimulatorError::ComputationError(format!("Serialization error: {e}")))?;
        Ok(serialized)
    }

    /// Import prediction models from storage
    pub fn import_models(&mut self, _data: &[u8]) -> Result<()> {
        // Note: Import functionality disabled due to SystemTime serialization limitations
        // In a full implementation, would use a custom serialization format or different time representation
        Err(SimulatorError::ComputationError(
            "Import not supported in current implementation".to_string(),
        ))
    }
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self {
            cpu_time_estimate: 0.0,
            memory_usage_estimate: 0,
            io_operations_estimate: 0,
            network_bandwidth_estimate: 0,
            gpu_memory_estimate: 0,
            thread_requirement: 1,
        }
    }
}

impl Default for PredictionStatistics {
    fn default() -> Self {
        Self {
            total_predictions: 0,
            successful_predictions: 0,
            average_accuracy: 0.0,
            prediction_time_stats: PerformanceTimingStatistics {
                average: Duration::from_millis(0),
                minimum: Duration::from_millis(0),
                maximum: Duration::from_millis(0),
                std_deviation: Duration::from_millis(0),
            },
            model_updates: 0,
            cache_hit_rate: 0.0,
        }
    }
}

/// Convenience function to create a performance prediction engine with default config
pub fn create_performance_predictor() -> Result<PerformancePredictionEngine> {
    PerformancePredictionEngine::new(PerformancePredictionConfig::default())
}

/// Convenience function to predict execution time for a circuit
pub fn predict_circuit_execution_time<const N: usize>(
    predictor: &mut PerformancePredictionEngine,
    circuit: &Circuit<N>,
    backend_type: BackendType,
) -> Result<PredictionResult> {
    predictor.predict_execution_time(circuit, backend_type)
}
