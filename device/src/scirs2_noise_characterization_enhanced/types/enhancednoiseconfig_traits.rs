//! # EnhancedNoiseConfig - Trait Implementations
//!
//! This module contains trait implementations for `EnhancedNoiseConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    AnalysisParameters, EnhancedNoiseConfig, NoiseCharacterizationConfig, NoiseModel,
    ReportingOptions, StatisticalMethod,
};

impl Default for EnhancedNoiseConfig {
    fn default() -> Self {
        Self {
            base_config: NoiseCharacterizationConfig::default(),
            enable_ml_analysis: true,
            enable_temporal_tracking: true,
            enable_spectral_analysis: true,
            enable_correlation_analysis: true,
            enable_predictive_modeling: true,
            enable_realtime_monitoring: true,
            noise_models: vec![
                NoiseModel::Depolarizing,
                NoiseModel::Dephasing,
                NoiseModel::AmplitudeDamping,
                NoiseModel::ThermalRelaxation,
                NoiseModel::CoherentError,
            ],
            statistical_methods: vec![
                StatisticalMethod::MaximumLikelihood,
                StatisticalMethod::BayesianInference,
                StatisticalMethod::SpectralDensity,
            ],
            analysis_parameters: AnalysisParameters::default(),
            reporting_options: ReportingOptions::default(),
        }
    }
}
