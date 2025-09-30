//! Core quantum anomaly detection functionality

use crate::error::{MLError, Result};
use scirs2_core::ndarray::{Array1, Array2};
use quantrs2_circuit::builder::Circuit;
use std::collections::{HashMap, VecDeque};

use super::config::*;
use super::metrics::*;
use super::preprocessing::*;

/// Trait for anomaly detection methods
pub trait AnomalyDetectorTrait {
    /// Train the detector on normal data
    fn fit(&mut self, data: &Array2<f64>) -> Result<()>;

    /// Detect anomalies in data
    fn detect(&self, data: &Array2<f64>) -> Result<AnomalyResult>;

    /// Update detector with new data (online learning)
    fn update(&mut self, data: &Array2<f64>, labels: Option<&Array1<i32>>) -> Result<()>;

    /// Get detector configuration
    fn get_config(&self) -> String;

    /// Get detector type
    fn get_type(&self) -> String;
}

/// Main quantum anomaly detector
pub struct QuantumAnomalyDetector {
    /// Configuration
    config: QuantumAnomalyConfig,

    /// Primary detection model
    primary_detector: Box<dyn AnomalyDetectorTrait>,

    /// Ensemble detectors
    ensemble_detectors: Vec<Box<dyn AnomalyDetectorTrait>>,

    /// Preprocessing pipeline
    preprocessor: DataPreprocessor,

    /// Real-time buffer for streaming detection
    realtime_buffer: Option<VecDeque<Array1<f64>>>,

    /// Training statistics
    training_stats: Option<TrainingStats>,

    /// Quantum circuits cache
    circuit_cache: HashMap<String, Circuit<16>>,

    /// Performance monitoring
    performance_monitor: PerformanceMonitor,
}

/// Performance monitoring
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Detection latencies
    latencies: VecDeque<f64>,

    /// Memory usage history
    memory_usage: VecDeque<f64>,

    /// Accuracy history (if ground truth available)
    accuracy_history: VecDeque<f64>,

    /// Quantum error rates
    quantum_error_rates: VecDeque<f64>,

    /// Detection latencies for test compatibility
    pub detection_latency: Vec<f64>,

    /// Throughput measurements
    pub throughput: Vec<f64>,

    /// Accuracy scores
    pub accuracy_scores: Vec<f64>,

    /// False positive rates
    pub false_positive_rate: Vec<f64>,

    /// Resource usage metrics
    pub resource_usage: Vec<f64>,
}

impl QuantumAnomalyDetector {
    /// Create a new quantum anomaly detector
    pub fn new(config: QuantumAnomalyConfig) -> Result<Self> {
        // Validate configuration
        if config.num_qubits == 0 {
            return Err(MLError::InvalidConfiguration(
                "Number of qubits must be greater than 0".to_string(),
            ));
        }

        if config.contamination < 0.0 || config.contamination > 1.0 {
            return Err(MLError::InvalidConfiguration(
                "Contamination must be between 0 and 1".to_string(),
            ));
        }

        // Create primary detector
        let primary_detector = Self::create_detector(&config.primary_method, &config)?;

        // Create ensemble detectors
        let mut ensemble_detectors = Vec::new();
        for method in &config.ensemble_methods {
            let detector = Self::create_detector(method, &config)?;
            ensemble_detectors.push(detector);
        }

        // Create preprocessor
        let preprocessor = DataPreprocessor::new(config.preprocessing.clone());

        // Initialize real-time buffer if configured
        let realtime_buffer = if let Some(realtime_config) = &config.realtime_config {
            Some(VecDeque::with_capacity(realtime_config.buffer_size))
        } else {
            None
        };

        // Initialize performance monitor
        let performance_monitor = PerformanceMonitor {
            latencies: VecDeque::new(),
            memory_usage: VecDeque::new(),
            accuracy_history: VecDeque::new(),
            quantum_error_rates: VecDeque::new(),
            detection_latency: Vec::new(),
            throughput: Vec::new(),
            accuracy_scores: Vec::new(),
            false_positive_rate: Vec::new(),
            resource_usage: Vec::new(),
        };

        Ok(Self {
            config,
            primary_detector,
            ensemble_detectors,
            preprocessor,
            realtime_buffer,
            training_stats: None,
            circuit_cache: HashMap::new(),
            performance_monitor,
        })
    }

    /// Create a detector based on the method configuration
    fn create_detector(
        method: &AnomalyDetectionMethod,
        config: &QuantumAnomalyConfig,
    ) -> Result<Box<dyn AnomalyDetectorTrait>> {
        use super::algorithms::*;

        match method {
            AnomalyDetectionMethod::QuantumIsolationForest { .. } => {
                Ok(Box::new(QuantumIsolationForest::new(config.clone())?))
            }
            AnomalyDetectionMethod::QuantumAutoencoder { .. } => {
                Ok(Box::new(QuantumAutoencoder::new(config.clone())?))
            }
            AnomalyDetectionMethod::QuantumOneClassSVM { .. } => {
                Ok(Box::new(QuantumOneClassSVM::new(config.clone())?))
            }
            AnomalyDetectionMethod::QuantumLOF { .. } => {
                Ok(Box::new(QuantumLOF::new(config.clone())?))
            }
            AnomalyDetectionMethod::QuantumDBSCAN { .. } => {
                Ok(Box::new(QuantumDBSCAN::new(config.clone())?))
            }
            AnomalyDetectionMethod::QuantumKMeansDetection { .. } => {
                Ok(Box::new(QuantumKMeansDetection::new(config.clone())?))
            }
            AnomalyDetectionMethod::QuantumNoveltyDetection { .. } => {
                Ok(Box::new(QuantumNoveltyDetection::new(config.clone())?))
            }
            AnomalyDetectionMethod::QuantumEnsemble { .. } => {
                Ok(Box::new(QuantumEnsemble::new(config.clone())?))
            }
        }
    }

    /// Train the anomaly detector
    pub fn fit(&mut self, data: &Array2<f64>) -> Result<()> {
        let start_time = std::time::Instant::now();

        // Preprocess data
        let processed_data = self.preprocessor.fit_transform(data)?;

        // Train primary detector
        self.primary_detector.fit(&processed_data)?;

        // Train ensemble detectors
        for detector in &mut self.ensemble_detectors {
            detector.fit(&processed_data)?;
        }

        // Update training statistics
        let training_time = start_time.elapsed().as_secs_f64();
        self.training_stats = Some(TrainingStats {
            training_time,
            n_training_samples: data.nrows(),
            feature_stats: self.compute_feature_stats(data)?,
            circuit_stats: CircuitStats {
                avg_depth: 0.0,
                avg_gates: 0.0,
                avg_execution_time: 0.0,
                success_rate: 1.0,
            },
        });

        Ok(())
    }

    /// Detect anomalies in new data
    pub fn detect(&self, data: &Array2<f64>) -> Result<AnomalyResult> {
        let start_time = std::time::Instant::now();

        // Preprocess data
        let processed_data = self.preprocessor.transform(data)?;

        // Get primary detection results
        let primary_result = self.primary_detector.detect(&processed_data)?;

        // Get ensemble results if available
        let mut ensemble_results = Vec::new();
        for detector in &self.ensemble_detectors {
            let result = detector.detect(&processed_data)?;
            ensemble_results.push(result);
        }

        // Combine results
        let final_result = self.combine_results(primary_result, ensemble_results)?;

        // Update performance monitoring
        let detection_time = start_time.elapsed().as_secs_f64();
        // Performance monitoring update would go here

        Ok(final_result)
    }

    /// Update detector with new data (online learning)
    pub fn update(&mut self, data: &Array2<f64>, labels: Option<&Array1<i32>>) -> Result<()> {
        // Preprocess data
        let processed_data = self.preprocessor.transform(data)?;

        // Update primary detector
        self.primary_detector.update(&processed_data, labels)?;

        // Update ensemble detectors
        for detector in &mut self.ensemble_detectors {
            detector.update(&processed_data, labels)?;
        }

        Ok(())
    }

    /// Get detector configuration
    pub fn get_config(&self) -> &QuantumAnomalyConfig {
        &self.config
    }

    /// Get training statistics
    pub fn get_training_stats(&self) -> Option<&TrainingStats> {
        self.training_stats.as_ref()
    }

    // Private helper methods

    fn compute_feature_stats(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        let n_features = data.ncols();
        let mut stats = Array2::zeros((n_features, 4)); // mean, std, min, max

        for i in 0..n_features {
            let col = data.column(i);
            let mean = col.mean().unwrap_or(0.0);
            let std = col.var(0.0).sqrt();
            let min = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            stats[[i, 0]] = mean;
            stats[[i, 1]] = std;
            stats[[i, 2]] = min;
            stats[[i, 3]] = max;
        }

        Ok(stats)
    }

    fn combine_results(
        &self,
        mut primary: AnomalyResult,
        ensemble: Vec<AnomalyResult>,
    ) -> Result<AnomalyResult> {
        // Combine method results from all detectors
        for ensemble_result in ensemble {
            for (method_name, method_result) in ensemble_result.method_results {
                primary.method_results.insert(method_name, method_result);
            }
        }

        Ok(primary)
    }
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new() -> Self {
        Self {
            latencies: VecDeque::new(),
            memory_usage: VecDeque::new(),
            accuracy_history: VecDeque::new(),
            quantum_error_rates: VecDeque::new(),
            detection_latency: Vec::new(),
            throughput: Vec::new(),
            accuracy_scores: Vec::new(),
            false_positive_rate: Vec::new(),
            resource_usage: Vec::new(),
        }
    }

    /// Record detection latency
    pub fn record_latency(&mut self, latency: f64) {
        self.latencies.push_back(latency);
        self.detection_latency.push(latency);

        // Keep only recent entries
        if self.latencies.len() > 1000 {
            self.latencies.pop_front();
        }
    }

    /// Record memory usage
    pub fn record_memory_usage(&mut self, usage: f64) {
        self.memory_usage.push_back(usage);
        self.resource_usage.push(usage);

        // Keep only recent entries
        if self.memory_usage.len() > 1000 {
            self.memory_usage.pop_front();
        }
    }

    /// Get average latency
    pub fn get_average_latency(&self) -> f64 {
        if self.latencies.is_empty() {
            0.0
        } else {
            self.latencies.iter().sum::<f64>() / self.latencies.len() as f64
        }
    }

    /// Get peak memory usage
    pub fn get_peak_memory_usage(&self) -> f64 {
        self.memory_usage.iter().fold(0.0, |a, &b| a.max(b))
    }
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}
