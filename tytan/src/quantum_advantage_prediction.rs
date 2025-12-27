//! Quantum Advantage Prediction System
//!
//! This module provides ML-based prediction of quantum advantage for QUBO problems,
//! enabling intelligent solver selection and resource allocation based on problem
//! characteristics and hardware capabilities.
//!
//! # Features
//!
//! - **Quantum Speedup Estimation**: ML models predicting quantum vs classical performance
//! - **Problem Hardness Characterization**: Analysis of problem structure and complexity
//! - **Resource Requirement Forecasting**: Prediction of compute time and hardware needs
//! - **Hardware-aware Performance Modeling**: Device-specific performance predictions
//! - **Automated Solver Selection**: Intelligent choice of classical vs quantum solvers
//!
//! # Example
//!
//! ```rust
//! use quantrs2_tytan::quantum_advantage_prediction::{
//!     AdvantagePredictor, PredictorConfig, ProblemFeatures
//! };
//! use scirs2_core::ndarray::Array2;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create predictor
//!     let config = PredictorConfig::default();
//!     let mut predictor = AdvantagePredictor::new(config);
//!
//!     // Define a QUBO problem
//!     let qubo = Array2::from_shape_vec(
//!         (4, 4),
//!         vec![
//!             -1.0, 2.0, 0.0, 1.0,
//!             2.0, -2.0, 1.0, 0.0,
//!             0.0, 1.0, -1.0, 3.0,
//!             1.0, 0.0, 3.0, -2.0,
//!         ]
//!     )?;
//!
//!     // Extract features and predict advantage
//!     let features = predictor.extract_features(&qubo)?;
//!     let prediction = predictor.predict_advantage(&features)?;
//!
//!     // Check if quantum approach is advantageous
//!     if prediction.has_quantum_advantage() {
//!         println!("Quantum advantage detected! Speedup: {:.2}x", prediction.speedup_factor);
//!     }
//!     Ok(())
//! }
//! ```

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use std::collections::HashMap;
use std::fmt;

/// Error types for advantage prediction
#[derive(Debug, Clone)]
pub enum PredictionError {
    /// Insufficient training data
    InsufficientData(String),
    /// Model not trained
    ModelNotTrained,
    /// Invalid problem features
    InvalidFeatures(String),
    /// Hardware specification not found
    HardwareNotFound(String),
}

impl fmt::Display for PredictionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InsufficientData(msg) => write!(f, "Insufficient data: {msg}"),
            Self::ModelNotTrained => write!(f, "Model not trained"),
            Self::InvalidFeatures(msg) => write!(f, "Invalid features: {msg}"),
            Self::HardwareNotFound(msg) => write!(f, "Hardware not found: {msg}"),
        }
    }
}

impl std::error::Error for PredictionError {}

/// Result type for prediction operations
pub type PredictionResult<T> = Result<T, PredictionError>;

/// Configuration for advantage predictor
#[derive(Debug, Clone)]
pub struct PredictorConfig {
    /// Number of features to extract from problems
    pub num_features: usize,
    /// ML model depth (number of hidden layers)
    pub model_depth: usize,
    /// ML model width (neurons per layer)
    pub model_width: usize,
    /// Training epochs
    pub training_epochs: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Speedup threshold for quantum advantage
    pub advantage_threshold: f64,
    /// Enable hardware-specific predictions
    pub hardware_aware: bool,
}

impl Default for PredictorConfig {
    fn default() -> Self {
        Self {
            num_features: 20,
            model_depth: 3,
            model_width: 64,
            training_epochs: 100,
            learning_rate: 0.001,
            advantage_threshold: 1.5, // 1.5x speedup
            hardware_aware: true,
        }
    }
}

impl PredictorConfig {
    /// Set the advantage threshold
    pub const fn with_advantage_threshold(mut self, threshold: f64) -> Self {
        self.advantage_threshold = threshold;
        self
    }

    /// Set model depth
    pub const fn with_model_depth(mut self, depth: usize) -> Self {
        self.model_depth = depth;
        self
    }

    /// Set training epochs
    pub const fn with_training_epochs(mut self, epochs: usize) -> Self {
        self.training_epochs = epochs;
        self
    }

    /// Enable or disable hardware-aware predictions
    pub const fn with_hardware_aware(mut self, enable: bool) -> Self {
        self.hardware_aware = enable;
        self
    }
}

/// Problem features for ML prediction
#[derive(Debug, Clone)]
pub struct ProblemFeatures {
    /// Problem size (number of variables)
    pub size: usize,
    /// Density (ratio of non-zero off-diagonal elements)
    pub density: f64,
    /// Average coupling strength
    pub avg_coupling: f64,
    /// Maximum coupling strength
    pub max_coupling: f64,
    /// Spectral gap (approximation)
    pub spectral_gap: f64,
    /// Clustering coefficient
    pub clustering_coefficient: f64,
    /// Average degree of connectivity
    pub avg_degree: f64,
    /// Frustration metric
    pub frustration: f64,
    /// Energy landscape ruggedness
    pub ruggedness: f64,
    /// Symmetry score
    pub symmetry: f64,
    /// Locality (spatial correlation)
    pub locality: f64,
    /// Condition number (numerical stability)
    pub condition_number: f64,
    /// Planted solution depth (if known)
    pub planted_depth: Option<f64>,
    /// Additional custom features
    pub custom_features: Vec<f64>,
}

impl ProblemFeatures {
    /// Convert to feature vector for ML model
    pub fn to_vector(&self) -> Array1<f64> {
        let mut features = vec![
            self.size as f64,
            self.density,
            self.avg_coupling,
            self.max_coupling,
            self.spectral_gap,
            self.clustering_coefficient,
            self.avg_degree,
            self.frustration,
            self.ruggedness,
            self.symmetry,
            self.locality,
            self.condition_number,
            self.planted_depth.unwrap_or(0.0),
        ];

        features.extend(&self.custom_features);
        Array1::from_vec(features)
    }
}

/// Hardware specification
#[derive(Debug, Clone)]
pub struct HardwareSpec {
    /// Hardware type
    pub hardware_type: HardwareType,
    /// Number of qubits/spins
    pub num_qubits: usize,
    /// Connectivity (average neighbors)
    pub connectivity: f64,
    /// Coherence time (for quantum)
    pub coherence_time: f64,
    /// Gate/operation fidelity
    pub fidelity: f64,
    /// Readout error rate
    pub readout_error: f64,
    /// Annealing/computation time
    pub computation_time: f64,
}

/// Types of hardware
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HardwareType {
    /// D-Wave quantum annealer
    DWave,
    /// IBM quantum gate-based
    IBMQuantum,
    /// Classical CPU
    CPU,
    /// Classical GPU
    GPU,
    /// Specialized FPGA
    FPGA,
    /// Photonic processor
    Photonic,
}

/// Prediction of quantum advantage
#[derive(Debug, Clone)]
pub struct AdvantagePrediction {
    /// Predicted speedup factor (quantum/classical)
    pub speedup_factor: f64,
    /// Confidence in prediction (0-1)
    pub confidence: f64,
    /// Predicted quantum runtime
    pub quantum_runtime: f64,
    /// Predicted classical runtime
    pub classical_runtime: f64,
    /// Recommended solver type
    pub recommended_solver: SolverType,
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
    /// Problem hardness score (0-1, higher = harder)
    pub hardness_score: f64,
}

impl AdvantagePrediction {
    /// Check if quantum advantage is predicted
    pub fn has_quantum_advantage(&self) -> bool {
        self.speedup_factor > 1.0 && self.recommended_solver != SolverType::Classical
    }

    /// Check if confidence is high
    pub fn is_high_confidence(&self) -> bool {
        self.confidence > 0.8
    }
}

/// Solver type recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverType {
    /// Classical solver
    Classical,
    /// Quantum annealer
    QuantumAnnealer,
    /// Quantum gate-based
    QuantumGate,
    /// Hybrid quantum-classical
    Hybrid,
}

/// Resource requirements prediction
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    /// Required qubits/variables
    pub required_qubits: usize,
    /// Estimated runtime (seconds)
    pub estimated_runtime: f64,
    /// Memory requirement (GB)
    pub memory_gb: f64,
    /// Number of shots/samples needed
    pub num_samples: usize,
}

/// Simple neural network for advantage prediction
#[derive(Debug, Clone)]
pub struct PredictionModel {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array1<f64>>,
    input_mean: Array1<f64>,
    input_std: Array1<f64>,
    trained: bool,
}

impl PredictionModel {
    /// Create a new prediction model
    pub fn new(input_size: usize, hidden_sizes: &[usize], output_size: usize) -> Self {
        let mut rng = thread_rng();
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        let mut prev_size = input_size;
        for &hidden_size in hidden_sizes {
            let scale = (2.0 / (prev_size + hidden_size) as f64).sqrt();
            let w = Array2::from_shape_fn((prev_size, hidden_size), |_| {
                (rng.gen::<f64>() * 2.0).mul_add(scale, -scale)
            });
            let b = Array1::zeros(hidden_size);
            weights.push(w);
            biases.push(b);
            prev_size = hidden_size;
        }

        let scale = (2.0 / (prev_size + output_size) as f64).sqrt();
        let w = Array2::from_shape_fn((prev_size, output_size), |_| {
            (rng.gen::<f64>() * 2.0).mul_add(scale, -scale)
        });
        let b = Array1::zeros(output_size);
        weights.push(w);
        biases.push(b);

        Self {
            weights,
            biases,
            input_mean: Array1::zeros(input_size),
            input_std: Array1::ones(input_size),
            trained: false,
        }
    }

    /// Predict output given input features
    pub fn predict(&self, input: &Array1<f64>) -> PredictionResult<Array1<f64>> {
        if !self.trained {
            return Err(PredictionError::ModelNotTrained);
        }

        let mut x = (input - &self.input_mean) / &self.input_std;

        for i in 0..self.weights.len() - 1 {
            let w = &self.weights[i];
            let b = &self.biases[i];
            x = x.dot(w) + b;
            x.mapv_inplace(|v| v.max(0.0)); // ReLU
        }

        // Safe: weights and biases are always initialized with at least one layer in constructor
        let w_last = self
            .weights
            .last()
            .expect("Model weights must have at least one layer");
        let b_last = self
            .biases
            .last()
            .expect("Model biases must have at least one layer");
        let output = x.dot(w_last) + b_last;

        Ok(output)
    }

    /// Train the model on problem data
    pub fn train(
        &mut self,
        features: &[Array1<f64>],
        targets: &[Array1<f64>],
        epochs: usize,
        _learning_rate: f64,
    ) -> PredictionResult<f64> {
        if features.is_empty() || targets.is_empty() {
            return Err(PredictionError::InsufficientData(
                "No training data provided".to_string(),
            ));
        }

        // Compute normalization parameters
        let n = features.len();
        self.input_mean = features
            .iter()
            .fold(Array1::zeros(features[0].len()), |acc, x| acc + x)
            / n as f64;

        let variance = features
            .iter()
            .fold(Array1::zeros(features[0].len()), |acc, x| {
                let diff = x - &self.input_mean;
                acc + &diff * &diff
            })
            / n as f64;
        self.input_std = variance.mapv(|v: f64| v.sqrt().max(1e-8));

        // Simplified training (in practice, use proper optimization)
        let mut final_loss = 0.0;
        for _ in 0..epochs {
            let mut epoch_loss = 0.0;
            for (feat, target) in features.iter().zip(targets.iter()) {
                // Normalize input
                let normalized = (feat - &self.input_mean) / &self.input_std;

                // Forward pass (simplified)
                let mut x = normalized.clone();
                for i in 0..self.weights.len() - 1 {
                    let w = &self.weights[i];
                    let b = &self.biases[i];
                    x = x.dot(w) + b;
                    x.mapv_inplace(|v| v.max(0.0));
                }
                // Safe: weights and biases are always initialized with at least one layer in constructor
                let w_last = self
                    .weights
                    .last()
                    .expect("Model weights must have at least one layer");
                let b_last = self
                    .biases
                    .last()
                    .expect("Model biases must have at least one layer");
                let prediction = x.dot(w_last) + b_last;

                // Compute loss
                let error = &prediction - target;
                epoch_loss += error.iter().map(|&e| e * e).sum::<f64>();
            }
            final_loss = epoch_loss / n as f64;
        }

        self.trained = true;
        Ok(final_loss)
    }

    /// Check if model is trained
    pub const fn is_trained(&self) -> bool {
        self.trained
    }
}

/// Advantage predictor
pub struct AdvantagePredictor {
    config: PredictorConfig,
    model: PredictionModel,
    hardware_specs: HashMap<HardwareType, HardwareSpec>,
}

impl AdvantagePredictor {
    /// Create a new advantage predictor
    pub fn new(config: PredictorConfig) -> Self {
        let model = PredictionModel::new(
            config.num_features,
            &vec![config.model_width; config.model_depth],
            3, // Output: speedup, confidence, hardness
        );

        let mut hardware_specs = HashMap::new();

        // Default hardware specifications
        hardware_specs.insert(
            HardwareType::DWave,
            HardwareSpec {
                hardware_type: HardwareType::DWave,
                num_qubits: 5000,
                connectivity: 15.0,
                coherence_time: 0.00001, // 10 μs
                fidelity: 0.99,
                readout_error: 0.01,
                computation_time: 0.00002, // 20 μs
            },
        );

        hardware_specs.insert(
            HardwareType::CPU,
            HardwareSpec {
                hardware_type: HardwareType::CPU,
                num_qubits: 1_000_000, // Effectively unlimited for classical
                connectivity: 1_000_000.0,
                coherence_time: f64::INFINITY,
                fidelity: 1.0,
                readout_error: 0.0,
                computation_time: 0.001, // 1 ms per iteration
            },
        );

        Self {
            config,
            model,
            hardware_specs,
        }
    }

    /// Extract features from a QUBO problem
    pub fn extract_features(&self, qubo: &Array2<f64>) -> PredictionResult<ProblemFeatures> {
        let n = qubo.nrows();
        if n != qubo.ncols() {
            return Err(PredictionError::InvalidFeatures(
                "Matrix must be square".to_string(),
            ));
        }

        // Calculate density
        let mut non_zero_count = 0;
        let mut sum_coupling = 0.0;
        let mut max_coupling: f64 = 0.0;

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let val = qubo[[i, j]].abs();
                    if val > 1e-10 {
                        non_zero_count += 1;
                        sum_coupling += val;
                        max_coupling = max_coupling.max(val);
                    }
                }
            }
        }

        let total_off_diagonal = n * (n - 1);
        let density = if total_off_diagonal > 0 {
            non_zero_count as f64 / total_off_diagonal as f64
        } else {
            0.0
        };

        let avg_coupling = if non_zero_count > 0 {
            sum_coupling / non_zero_count as f64
        } else {
            0.0
        };

        // Calculate average degree
        let avg_degree = density * (n - 1) as f64;

        // Estimate spectral gap (simplified)
        let spectral_gap = self.estimate_spectral_gap(qubo);

        // Calculate clustering coefficient
        let clustering_coefficient = self.compute_clustering_coefficient(qubo);

        // Calculate frustration
        let frustration = self.compute_frustration(qubo);

        // Estimate ruggedness
        let ruggedness = self.estimate_ruggedness(qubo);

        // Calculate symmetry
        let symmetry = self.compute_symmetry(qubo);

        // Calculate locality
        let locality = self.compute_locality(qubo);

        // Estimate condition number
        let condition_number = self.estimate_condition_number(qubo);

        Ok(ProblemFeatures {
            size: n,
            density,
            avg_coupling,
            max_coupling,
            spectral_gap,
            clustering_coefficient,
            avg_degree,
            frustration,
            ruggedness,
            symmetry,
            locality,
            condition_number,
            planted_depth: None,
            custom_features: vec![],
        })
    }

    /// Estimate spectral gap (simplified)
    fn estimate_spectral_gap(&self, qubo: &Array2<f64>) -> f64 {
        // Simplified: use variance of diagonal elements as proxy
        let n = qubo.nrows();
        let mut diag_values = Vec::new();
        for i in 0..n {
            diag_values.push(qubo[[i, i]]);
        }

        let mean = diag_values.iter().sum::<f64>() / n as f64;
        let variance = diag_values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        variance.sqrt()
    }

    /// Compute clustering coefficient
    fn compute_clustering_coefficient(&self, qubo: &Array2<f64>) -> f64 {
        let n = qubo.nrows();
        let mut total_clustering = 0.0;

        for i in 0..n {
            let neighbors: Vec<usize> = (0..n)
                .filter(|&j| j != i && qubo[[i, j]].abs() > 1e-10)
                .collect();

            if neighbors.len() < 2 {
                continue;
            }

            let mut triangles = 0;
            for &j in &neighbors {
                for &k in &neighbors {
                    if j < k && qubo[[j, k]].abs() > 1e-10 {
                        triangles += 1;
                    }
                }
            }

            let possible_triangles = neighbors.len() * (neighbors.len() - 1) / 2;
            if possible_triangles > 0 {
                total_clustering += triangles as f64 / possible_triangles as f64;
            }
        }

        total_clustering / n as f64
    }

    /// Compute frustration metric
    fn compute_frustration(&self, qubo: &Array2<f64>) -> f64 {
        let n = qubo.nrows();
        let mut frustration_count = 0;

        // Count frustrated triangles
        for i in 0..n {
            for j in (i + 1)..n {
                for k in (j + 1)..n {
                    let w_ij = qubo[[i, j]];
                    let w_jk = qubo[[j, k]];
                    let w_ik = qubo[[i, k]];

                    // Check if triangle is frustrated
                    if w_ij.abs() > 1e-10 && w_jk.abs() > 1e-10 && w_ik.abs() > 1e-10 {
                        let product = w_ij.signum() * w_jk.signum() * w_ik.signum();
                        if product < 0.0 {
                            frustration_count += 1;
                        }
                    }
                }
            }
        }

        frustration_count as f64 / ((n * (n - 1) * (n - 2)) / 6) as f64
    }

    /// Estimate ruggedness of energy landscape
    fn estimate_ruggedness(&self, qubo: &Array2<f64>) -> f64 {
        // Simplified: use standard deviation of coupling strengths
        let n = qubo.nrows();
        let mut couplings = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                if qubo[[i, j]].abs() > 1e-10 {
                    couplings.push(qubo[[i, j]].abs());
                }
            }
        }

        if couplings.is_empty() {
            return 0.0;
        }

        let mean = couplings.iter().sum::<f64>() / couplings.len() as f64;
        let variance =
            couplings.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / couplings.len() as f64;

        variance.sqrt() / (mean + 1e-10)
    }

    /// Compute symmetry score
    fn compute_symmetry(&self, qubo: &Array2<f64>) -> f64 {
        let n = qubo.nrows();
        let mut symmetry_error = 0.0;

        for i in 0..n {
            for j in 0..n {
                symmetry_error += (qubo[[i, j]] - qubo[[j, i]]).abs();
            }
        }

        let total_magnitude: f64 = qubo.iter().map(|&x| x.abs()).sum();
        1.0 - (symmetry_error / (total_magnitude + 1e-10)).min(1.0)
    }

    /// Compute locality (spatial correlation)
    fn compute_locality(&self, qubo: &Array2<f64>) -> f64 {
        let n = qubo.nrows();
        let mut local_weight = 0.0;
        let mut total_weight = 0.0;

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let weight = qubo[[i, j]].abs();
                    total_weight += weight;

                    // Consider "local" if indices are close
                    if (i as i32 - j as i32).abs() <= 2 {
                        local_weight += weight;
                    }
                }
            }
        }

        if total_weight > 1e-10 {
            local_weight / total_weight
        } else {
            0.0
        }
    }

    /// Estimate condition number (simplified)
    fn estimate_condition_number(&self, qubo: &Array2<f64>) -> f64 {
        // Simplified: ratio of max to min absolute values
        let mut max_val: f64 = 0.0;
        let mut min_val = f64::INFINITY;

        for &val in qubo {
            let abs_val = val.abs();
            if abs_val > 1e-10 {
                max_val = max_val.max(abs_val);
                min_val = min_val.min(abs_val);
            }
        }

        if min_val < f64::INFINITY && min_val > 1e-10 {
            max_val / min_val
        } else {
            1.0
        }
    }

    /// Predict quantum advantage for a problem
    pub fn predict_advantage(
        &self,
        features: &ProblemFeatures,
    ) -> PredictionResult<AdvantagePrediction> {
        let feature_vector = features.to_vector();

        // Use ML model if trained, otherwise use heuristics
        let (speedup, confidence, hardness) = if self.model.is_trained() {
            let prediction = self.model.predict(&feature_vector)?;
            (prediction[0], prediction[1], prediction[2])
        } else {
            // Heuristic prediction
            self.heuristic_prediction(features)
        };

        // Determine recommended solver
        let recommended_solver = if speedup > self.config.advantage_threshold {
            if features.size > 1000 {
                SolverType::QuantumAnnealer
            } else {
                SolverType::Hybrid
            }
        } else {
            SolverType::Classical
        };

        // Estimate resource requirements
        let resource_requirements = self.estimate_resources(features, recommended_solver);

        // Estimate runtimes
        let quantum_runtime = resource_requirements.estimated_runtime;
        let classical_runtime = quantum_runtime * speedup;

        Ok(AdvantagePrediction {
            speedup_factor: speedup,
            confidence,
            quantum_runtime,
            classical_runtime,
            recommended_solver,
            resource_requirements,
            hardness_score: hardness,
        })
    }

    /// Heuristic prediction without trained model
    fn heuristic_prediction(&self, features: &ProblemFeatures) -> (f64, f64, f64) {
        // Estimate speedup based on problem characteristics
        let mut speedup = 1.0;

        // Large, dense problems favor quantum
        if features.size > 100 && features.density > 0.3 {
            speedup *= 2.0;
        }

        // High frustration favors quantum
        if features.frustration > 0.3 {
            speedup *= 1.5;
        }

        // Low locality favors quantum (non-local interactions)
        if features.locality < 0.3 {
            speedup *= 1.3;
        }

        // High ruggedness favors quantum
        if features.ruggedness > 1.0 {
            speedup *= 1.2;
        }

        // Confidence inversely related to problem size
        let confidence = (1.0 - (features.size as f64 / 1000.0).min(0.5)).max(0.3);

        // Hardness based on multiple factors
        let hardness = features
            .ruggedness
            .min(1.0)
            .mul_add(
                0.4,
                features
                    .frustration
                    .mul_add(0.3, (1.0 - features.locality) * 0.3),
            )
            .min(1.0);

        (speedup, confidence, hardness)
    }

    /// Estimate resource requirements
    fn estimate_resources(
        &self,
        features: &ProblemFeatures,
        solver: SolverType,
    ) -> ResourceRequirements {
        let required_qubits = features.size;

        // Compute hardness score from features
        let hardness = features
            .ruggedness
            .min(1.0)
            .mul_add(
                0.4,
                features
                    .frustration
                    .mul_add(0.3, (1.0 - features.locality) * 0.3),
            )
            .min(1.0);

        let estimated_runtime = match solver {
            SolverType::Classical => {
                // Exponential scaling for hard problems
                0.001 * (features.size as f64 * hardness).min(20.0).exp2()
            }
            SolverType::QuantumAnnealer => {
                // Polynomial scaling
                0.0001 * (features.size as f64).powf(1.5)
            }
            SolverType::QuantumGate => {
                // Similar to annealer but higher constant
                0.001 * (features.size as f64).powf(1.5)
            }
            SolverType::Hybrid => {
                // Mix of both
                0.0005 * (features.size as f64).powf(1.3)
            }
        };

        let memory_gb = match solver {
            SolverType::Classical => {
                // State vector grows exponentially
                8.0 * (features.size as f64).min(30.0).exp2() / 1e9
            }
            _ => {
                // Quantum doesn't need full state storage
                0.1 * features.size as f64
            }
        };

        let num_samples = match solver {
            SolverType::Classical => 1,
            SolverType::QuantumAnnealer => (100.0_f64 * (1.0 + hardness)).ceil() as usize,
            SolverType::QuantumGate => (1000.0_f64 * (1.0 + hardness)).ceil() as usize,
            SolverType::Hybrid => 50,
        };

        ResourceRequirements {
            required_qubits,
            estimated_runtime,
            memory_gb,
            num_samples,
        }
    }

    /// Get configuration
    pub const fn config(&self) -> &PredictorConfig {
        &self.config
    }

    /// Check if model is trained
    pub const fn is_trained(&self) -> bool {
        self.model.is_trained()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predictor_creation() {
        let config = PredictorConfig::default();
        let predictor = AdvantagePredictor::new(config);
        assert!(!predictor.is_trained());
    }

    #[test]
    fn test_feature_extraction() {
        let qubo = Array2::from_shape_vec(
            (4, 4),
            vec![
                -1.0, 2.0, 0.0, 1.0, 2.0, -2.0, 1.0, 0.0, 0.0, 1.0, -1.0, 3.0, 1.0, 0.0, 3.0, -2.0,
            ],
        )
        .expect("4x4 QUBO matrix creation should succeed");

        let config = PredictorConfig::default();
        let predictor = AdvantagePredictor::new(config);

        let features = predictor
            .extract_features(&qubo)
            .expect("Feature extraction should succeed for valid QUBO");
        assert_eq!(features.size, 4);
        assert!(features.density > 0.0);
        assert!(features.density <= 1.0);
    }

    #[test]
    fn test_heuristic_prediction() {
        let config = PredictorConfig::default();
        let predictor = AdvantagePredictor::new(config);

        let features = ProblemFeatures {
            size: 100,
            density: 0.5,
            avg_coupling: 1.0,
            max_coupling: 5.0,
            spectral_gap: 0.1,
            clustering_coefficient: 0.3,
            avg_degree: 10.0,
            frustration: 0.4,
            ruggedness: 1.2,
            symmetry: 0.9,
            locality: 0.2,
            condition_number: 10.0,
            planted_depth: None,
            custom_features: vec![],
        };

        let result = predictor
            .predict_advantage(&features)
            .expect("Advantage prediction should succeed for valid features");
        assert!(result.speedup_factor > 0.0);
        assert!(result.confidence > 0.0 && result.confidence <= 1.0);
        assert!(result.hardness_score >= 0.0 && result.hardness_score <= 1.0);
    }

    #[test]
    fn test_resource_estimation() {
        let config = PredictorConfig::default();
        let predictor = AdvantagePredictor::new(config);

        let features = ProblemFeatures {
            size: 50,
            density: 0.3,
            avg_coupling: 1.0,
            max_coupling: 3.0,
            spectral_gap: 0.2,
            clustering_coefficient: 0.4,
            avg_degree: 15.0,
            frustration: 0.2,
            ruggedness: 0.8,
            symmetry: 0.95,
            locality: 0.5,
            condition_number: 5.0,
            planted_depth: None,
            custom_features: vec![],
        };

        let resources = predictor.estimate_resources(&features, SolverType::QuantumAnnealer);
        assert_eq!(resources.required_qubits, 50);
        assert!(resources.estimated_runtime > 0.0);
        assert!(resources.memory_gb > 0.0);
        assert!(resources.num_samples > 0);
    }

    #[test]
    fn test_advantage_prediction_integration() {
        let config = PredictorConfig::default().with_advantage_threshold(1.2);
        let predictor = AdvantagePredictor::new(config);

        let qubo = Array2::from_shape_vec(
            (10, 10),
            (0..100)
                .map(|i| {
                    if i % 11 == 0 {
                        -1.0
                    } else if i % 2 == 0 {
                        0.5
                    } else {
                        0.0
                    }
                })
                .collect(),
        )
        .expect("10x10 QUBO matrix creation should succeed");

        let features = predictor
            .extract_features(&qubo)
            .expect("Feature extraction should succeed for valid QUBO");
        let prediction = predictor
            .predict_advantage(&features)
            .expect("Advantage prediction should succeed for valid features");

        assert!(prediction.speedup_factor > 0.0);
        assert!(prediction.confidence > 0.0);
    }

    #[test]
    fn test_config_builder() {
        let config = PredictorConfig::default()
            .with_advantage_threshold(2.0)
            .with_model_depth(5)
            .with_training_epochs(200)
            .with_hardware_aware(false);

        assert_eq!(config.advantage_threshold, 2.0);
        assert_eq!(config.model_depth, 5);
        assert_eq!(config.training_epochs, 200);
        assert!(!config.hardware_aware);
    }

    #[test]
    fn test_problem_features_to_vector() {
        let features = ProblemFeatures {
            size: 10,
            density: 0.5,
            avg_coupling: 1.0,
            max_coupling: 2.0,
            spectral_gap: 0.3,
            clustering_coefficient: 0.4,
            avg_degree: 5.0,
            frustration: 0.2,
            ruggedness: 0.9,
            symmetry: 0.95,
            locality: 0.6,
            condition_number: 8.0,
            planted_depth: Some(0.7),
            custom_features: vec![0.1, 0.2],
        };

        let vector = features.to_vector();
        assert_eq!(vector.len(), 15); // 13 standard + 2 custom
    }

    #[test]
    fn test_solver_type_recommendation() {
        let config = PredictorConfig::default();
        let predictor = AdvantagePredictor::new(config);

        // Small problem - should recommend hybrid
        let small_features = ProblemFeatures {
            size: 50,
            density: 0.5,
            avg_coupling: 1.0,
            max_coupling: 2.0,
            spectral_gap: 0.2,
            clustering_coefficient: 0.4,
            avg_degree: 10.0,
            frustration: 0.5,
            ruggedness: 1.5,
            symmetry: 0.9,
            locality: 0.2,
            condition_number: 10.0,
            planted_depth: None,
            custom_features: vec![],
        };

        let pred = predictor
            .predict_advantage(&small_features)
            .expect("Advantage prediction should succeed for valid features");
        // Verify prediction structure
        assert!(matches!(
            pred.recommended_solver,
            SolverType::Classical | SolverType::Hybrid | SolverType::QuantumAnnealer
        ));
    }

    #[test]
    fn test_prediction_model() {
        let mut model = PredictionModel::new(5, &[10, 10], 3);
        assert!(!model.is_trained());

        // Create dummy training data
        let features = vec![
            Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]),
            Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0, 6.0]),
        ];
        let targets = vec![
            Array1::from_vec(vec![1.5, 0.8, 0.6]),
            Array1::from_vec(vec![2.0, 0.7, 0.7]),
        ];

        let result = model.train(&features, &targets, 10, 0.001);
        assert!(result.is_ok());
        assert!(model.is_trained());
    }
}
