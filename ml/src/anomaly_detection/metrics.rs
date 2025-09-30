//! Metrics and results for quantum anomaly detection

use super::config::TimeSeriesAnomalyType;
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Anomaly detection results
#[derive(Debug, Clone)]
pub struct AnomalyResult {
    /// Anomaly scores for each sample
    pub anomaly_scores: Array1<f64>,

    /// Binary anomaly labels (1 for anomaly, 0 for normal)
    pub anomaly_labels: Array1<i32>,

    /// Confidence scores
    pub confidence_scores: Array1<f64>,

    /// Explanation scores for each feature
    pub feature_importance: Array2<f64>,

    /// Method-specific results
    pub method_results: HashMap<String, MethodSpecificResult>,

    /// Performance metrics
    pub metrics: AnomalyMetrics,

    /// Processing statistics
    pub processing_stats: ProcessingStats,
}

/// Method-specific results
#[derive(Debug, Clone)]
pub enum MethodSpecificResult {
    IsolationForest {
        path_lengths: Array1<f64>,
        tree_depths: Array1<f64>,
    },
    Autoencoder {
        reconstruction_errors: Array1<f64>,
        latent_representations: Array2<f64>,
    },
    OneClassSVM {
        support_vectors: Array2<f64>,
        decision_function: Array1<f64>,
    },
    Clustering {
        cluster_assignments: Array1<usize>,
        cluster_distances: Array1<f64>,
    },
    LOF {
        local_outlier_factors: Array1<f64>,
        reachability_distances: Array1<f64>,
    },
    DBSCAN {
        cluster_labels: Array1<i32>,
        core_sample_indices: Vec<usize>,
    },
}

/// Anomaly detection metrics
#[derive(Debug, Clone)]
pub struct AnomalyMetrics {
    /// Area under ROC curve
    pub auc_roc: f64,

    /// Area under precision-recall curve
    pub auc_pr: f64,

    /// Precision at given contamination level
    pub precision: f64,

    /// Recall at given contamination level
    pub recall: f64,

    /// F1 score
    pub f1_score: f64,

    /// False positive rate
    pub false_positive_rate: f64,

    /// False negative rate
    pub false_negative_rate: f64,

    /// Matthews correlation coefficient
    pub mcc: f64,

    /// Balanced accuracy
    pub balanced_accuracy: f64,

    /// Quantum-specific metrics
    pub quantum_metrics: QuantumAnomalyMetrics,
}

/// Quantum-specific anomaly metrics
#[derive(Debug, Clone)]
pub struct QuantumAnomalyMetrics {
    /// Quantum advantage factor
    pub quantum_advantage: f64,

    /// Entanglement utilization
    pub entanglement_utilization: f64,

    /// Circuit depth efficiency
    pub circuit_efficiency: f64,

    /// Quantum error rate
    pub quantum_error_rate: f64,

    /// Coherence time utilization
    pub coherence_utilization: f64,
}

/// Processing statistics
#[derive(Debug, Clone)]
pub struct ProcessingStats {
    /// Total processing time (seconds)
    pub total_time: f64,

    /// Quantum processing time (seconds)
    pub quantum_time: f64,

    /// Classical processing time (seconds)
    pub classical_time: f64,

    /// Memory usage (MB)
    pub memory_usage: f64,

    /// Number of quantum circuit executions
    pub quantum_executions: usize,

    /// Average circuit depth
    pub avg_circuit_depth: f64,
}

/// Time series anomaly point
#[derive(Debug, Clone)]
pub struct TimeSeriesAnomalyPoint {
    /// Timestamp index
    pub timestamp: usize,

    /// Anomaly score
    pub score: f64,

    /// Anomaly type
    pub anomaly_type: TimeSeriesAnomalyType,

    /// Seasonal context
    pub seasonal_context: Option<super::config::SeasonalContext>,

    /// Trend context
    pub trend_context: Option<super::config::TrendContext>,
}

/// Training statistics
#[derive(Debug, Clone)]
pub struct TrainingStats {
    /// Training time
    pub training_time: f64,

    /// Number of training samples
    pub n_training_samples: usize,

    /// Feature statistics
    pub feature_stats: Array2<f64>, // mean, std, min, max per feature

    /// Quantum circuit statistics
    pub circuit_stats: CircuitStats,
}

/// Quantum circuit statistics
#[derive(Debug, Clone)]
pub struct CircuitStats {
    /// Average circuit depth
    pub avg_depth: f64,

    /// Average number of gates
    pub avg_gates: f64,

    /// Average execution time
    pub avg_execution_time: f64,

    /// Circuit success rate
    pub success_rate: f64,
}
