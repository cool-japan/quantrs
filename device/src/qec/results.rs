//! Quantum Error Correction Result Types
//!
//! This module defines the result types returned by QEC operations:
//! - Circuit correction results with performance metrics
//! - Syndrome analysis results with pattern recognition data
//! - Mitigation results for various error mitigation strategies
//! - Statistical analysis results for QEC performance
//! - ZNE (Zero-Noise Extrapolation) results

use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use quantrs2_circuit::prelude::Circuit;
use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};

use super::{
    config::{SpatialPattern, TemporalPattern},
    QECStrategy,
};

/// Comprehensive result of circuit error correction
#[derive(Debug, Clone)]
pub struct CorrectedCircuitResult<const N: usize> {
    pub original_circuit: Circuit<N>,
    pub corrected_circuit: Circuit<N>,
    pub applied_strategy: QECStrategy,
    pub syndrome_data: SyndromeAnalysisResult,
    pub mitigation_data: MitigationResult<N>,
    pub zne_data: Option<ZNEResult<N>>,
    pub correction_performance: CorrectionPerformance,
    pub statistical_analysis: StatisticalAnalysisResult,
}

/// Performance metrics for error correction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionPerformance {
    pub total_time: Duration,
    pub fidelity_improvement: f64,
    pub resource_overhead: f64,
    pub confidence_score: f64,
}

/// Analysis of error patterns in the quantum system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorPatternAnalysis {
    pub temporal_patterns: Vec<TemporalPattern>,
    pub spatial_patterns: Vec<SpatialPattern>,
    pub environmental_correlations: HashMap<String, f64>,
    pub ml_predictions: Vec<PredictedPattern>,
    pub confidence_score: f64,
    pub last_updated: SystemTime,
}

// TemporalPattern and SpatialPattern are imported from config

/// ML-predicted error pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictedPattern {
    pub pattern_type: String,
    pub probability: f64,
    pub time_horizon: Duration,
    pub affected_components: Vec<String>,
}

/// Syndrome measurements data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyndromeMeasurements {
    pub syndrome_bits: Vec<bool>,
    pub detected_errors: Vec<usize>,
    pub measurement_fidelity: f64,
    pub measurement_time: Duration,
    pub raw_measurements: HashMap<String, f64>,
}

/// Result of syndrome detection and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyndromeAnalysisResult {
    pub syndrome_measurements: SyndromeMeasurements,
    pub pattern_recognition: Option<PatternRecognitionResult>,
    pub statistical_analysis: Option<SyndromeStatistics>,
    pub historical_correlation: HistoricalCorrelation,
    pub detection_confidence: f64,
    pub timestamp: SystemTime,
}

/// Pattern recognition result from ML models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternRecognitionResult {
    pub recognized_patterns: Vec<String>,
    pub pattern_confidence: HashMap<String, f64>,
    pub ml_model_used: String,
    pub prediction_time: Duration,
}

/// Statistical analysis of syndromes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyndromeStatistics {
    pub error_rate_statistics: HashMap<String, f64>,
    pub distribution_analysis: String,
    pub confidence_intervals: HashMap<String, (f64, f64)>,
    pub statistical_tests: HashMap<String, f64>,
}

/// Correlation with historical syndrome patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalCorrelation {
    pub similarity_score: f64,
    pub matching_patterns: Vec<String>,
    pub temporal_correlation: f64,
    pub deviation_analysis: HashMap<String, f64>,
}

/// Result of error mitigation strategies
#[derive(Debug, Clone)]
pub struct MitigationResult<const N: usize> {
    pub circuit: Circuit<N>,
    pub applied_corrections: Vec<String>,
    pub resource_overhead: f64,
    pub effectiveness_score: f64,
    pub confidence_score: f64,
    pub mitigation_time: SystemTime,
}

/// Result of gate-level mitigation
#[derive(Debug, Clone)]
pub struct GateMitigationResult<const N: usize> {
    pub circuit: Circuit<N>,
    pub corrections: Vec<String>,
    pub resource_overhead: f64,
}

/// Result of symmetry verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymmetryVerificationResult {
    pub corrections: Vec<String>,
    pub overhead: f64,
}

/// Result of virtual distillation
#[derive(Debug, Clone)]
pub struct VirtualDistillationResult<const N: usize> {
    pub circuit: Circuit<N>,
    pub corrections: Vec<String>,
    pub overhead: f64,
}

/// Zero-Noise Extrapolation (ZNE) result
#[derive(Debug, Clone)]
pub struct ZNEResult<const N: usize> {
    pub original_circuit: Circuit<N>,
    pub scaled_circuits: Vec<f64>,
    pub extrapolated_result: HashMap<String, usize>,
    pub richardson_result: Option<HashMap<String, usize>>,
    pub statistical_confidence: f64,
    pub zne_overhead: f64,
}

/// Result of readout error correction
#[derive(Debug, Clone)]
pub struct ReadoutCorrectedResult<const N: usize> {
    pub circuit: Circuit<N>,
    pub correction_matrix: Array2<f64>,
    pub corrected_counts: HashMap<String, usize>,
    pub fidelity_improvement: f64,
    pub resource_overhead: f64,
    pub confidence_score: f64,
}

/// Statistical analysis of QEC performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysisResult {
    pub mean_success_rate: f64,
    pub std_success_rate: f64,
    pub trend_analysis: TrendAnalysisData,
    pub correlation_analysis: CorrelationAnalysisData,
    pub prediction_accuracy: f64,
    pub confidence_interval: (f64, f64),
    pub sample_size: usize,
    pub last_updated: SystemTime,
}

/// Trend analysis data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysisData {
    pub trend_direction: String,
    pub trend_strength: f64,
    pub confidence_level: f64,
}

/// Correlation analysis data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationAnalysisData {
    pub correlation_matrix: Array2<f64>,
    pub significant_correlations: Vec<(String, String, f64)>,
}
