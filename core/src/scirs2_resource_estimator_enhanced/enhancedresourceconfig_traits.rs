//! # EnhancedResourceConfig - Trait Implementations
//!
//! This module contains trait implementations for `EnhancedResourceConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::parallel_ops_stubs::*;
use crate::resource_estimator::{
    ErrorCorrectionCode, EstimationMode, HardwarePlatform, QuantumGate, ResourceEstimationConfig,
};

use super::types::{
    AnalysisDepth, CloudPlatform, EnhancedResourceConfig, OptimizationObjective, ReportFormat,
};

impl Default for EnhancedResourceConfig {
    fn default() -> Self {
        Self {
            base_config: ResourceEstimationConfig::default(),
            enable_ml_prediction: true,
            enable_cost_analysis: true,
            enable_optimization_strategies: true,
            enable_comparative_analysis: true,
            enable_realtime_tracking: true,
            enable_visual_representation: true,
            enable_hardware_recommendations: true,
            enable_scaling_predictions: true,
            cloud_platforms: vec![
                CloudPlatform::IBMQ,
                CloudPlatform::AzureQuantum,
                CloudPlatform::AmazonBraket,
            ],
            optimization_objectives: vec![
                OptimizationObjective::MinimizeTime,
                OptimizationObjective::MinimizeQubits,
                OptimizationObjective::MinimizeCost,
            ],
            analysis_depth: AnalysisDepth::Comprehensive,
            custom_constraints: Vec::new(),
            export_formats: vec![ReportFormat::JSON, ReportFormat::HTML, ReportFormat::PDF],
        }
    }
}
