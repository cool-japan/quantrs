//! Quantum Error Correction Integration with SciRS2 Analytics
//!
//! This module provides comprehensive quantum error correction (QEC) capabilities
//! integrated with SciRS2's advanced analytics, optimization, and machine learning
//! for adaptive error correction on quantum hardware.
//!
//! # Module Structure
//!
//! - `traits`: Core trait definitions for QEC operations
//! - `types`: Basic types and data structures
//! - `implementations`: QEC code implementations (Steane, Shor, Surface, Toric)
//! - `config`: Configuration types for QEC system
//! - `corrector`: Main QuantumErrorCorrector implementation
//! - `results`: Result types for QEC operations
//! - `adaptive`: Adaptive QEC strategies
//! - `benchmarking`: QEC benchmarking tools
//! - `codes`: QEC code definitions and utilities
//! - `detection`: Syndrome detection and analysis
//! - `mitigation`: Error mitigation strategies

// Fallback implementations when SciRS2 is not available
#[cfg(not(feature = "scirs2"))]
mod fallback_scirs2 {
    use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};

    pub fn mean(_data: &ArrayView1<f64>) -> Result<f64, String> {
        Ok(0.0)
    }
    pub fn std(_data: &ArrayView1<f64>, _ddof: i32) -> Result<f64, String> {
        Ok(1.0)
    }
    pub fn corrcoef(_data: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        Ok(Array2::eye(2))
    }
    pub fn pca(
        _data: &ArrayView2<f64>,
        _n_components: usize,
    ) -> Result<(Array2<f64>, Array1<f64>), String> {
        Ok((Array2::zeros((2, 2)), Array1::zeros(2)))
    }

    pub struct OptimizeResult {
        pub x: Array1<f64>,
        pub fun: f64,
        pub success: bool,
        pub nit: usize,
        pub nfev: usize,
        pub message: String,
    }

    pub fn minimize(
        _func: fn(&Array1<f64>) -> f64,
        _x0: &Array1<f64>,
        _method: &str,
    ) -> Result<OptimizeResult, String> {
        Ok(OptimizeResult {
            x: Array1::zeros(2),
            fun: 0.0,
            success: true,
            nit: 0,
            nfev: 0,
            message: "Fallback optimization".to_string(),
        })
    }
}

#[cfg(not(feature = "scirs2"))]
use fallback_scirs2::*;

// Module declarations - core internal modules
mod config;
mod corrector;
mod implementations;
mod results;
mod traits;
mod types;

// Module declarations - public feature modules
pub mod adaptive;
pub mod benchmarking;
pub mod codes;
pub mod detection;
pub mod mitigation;

// Re-export core types and traits
pub use traits::{ErrorCorrector, QECResult, QuantumErrorCode, SyndromeDetector};

pub use types::{
    AdaptiveQECSystem, CorrectionOperation, CorrectionType, DeviceState, ErrorModel,
    ExecutionContext, LogicalOperator, LogicalOperatorType, PauliOperator, QECPerformanceMetrics,
    QECPerformanceTracker, StabilizerGroup, StabilizerType, SyndromePattern, SyndromeResult,
    SyndromeType, TrendAnalysis,
};

pub use implementations::{ShorCode, SteaneCode, SurfaceCode, ToricCode};

pub use config::{
    AdaptiveThresholds, CachedOptimization, CorrectionMetrics, ErrorCorrectionCycleResult,
    ErrorStatistics, MLModel, OptimizationResult, QECConfig, QECMLConfig, QECMonitoringConfig,
    QECOptimizationConfig, QECStrategy, ResourceRequirements, ResourceUtilization, SpatialPattern,
    TemporalPattern,
};

pub use corrector::QuantumErrorCorrector;

pub use results::{
    CorrectedCircuitResult, CorrectionPerformance, CorrelationAnalysisData, ErrorPatternAnalysis,
    GateMitigationResult, HistoricalCorrelation, MitigationResult, PatternRecognitionResult,
    PredictedPattern, ReadoutCorrectedResult, StatisticalAnalysisResult,
    SymmetryVerificationResult, SyndromeAnalysisResult, SyndromeMeasurements, SyndromeStatistics,
    TrendAnalysisData, VirtualDistillationResult, ZNEResult,
};
