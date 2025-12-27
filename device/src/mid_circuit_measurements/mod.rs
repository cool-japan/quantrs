//! Mid-circuit measurements module
//!
//! This module provides comprehensive support for mid-circuit measurements in quantum circuits,
//! including advanced analytics, machine learning optimization, and adaptive learning capabilities.

pub mod analytics;
pub mod config;
pub mod executor;
pub mod fallback;
pub mod ml;
pub mod monitoring;
pub mod results;

// Re-export main types and interfaces
pub use config::{
    AdaptiveConfig, AdvancedAnalyticsConfig, FeatureEngineeringConfig, HardwareOptimizations,
    MLModelType, MLOptimizationConfig, MLTrainingConfig, MidCircuitConfig, OnlineLearningConfig,
    PredictionConfig, TimeSeriesConfig, UncertaintyConfig, ValidationConfig,
};

pub use results::{
    AdaptiveLearningInsights, AdvancedAnalyticsResults, ErrorAnalysis, ExecutionStats,
    MeasurementEvent, MeasurementPredictionResults, MeasurementType, MidCircuitCapabilities,
    MidCircuitExecutionResult, OptimizationRecommendations, PerformanceMetrics, StorageLocation,
    TimingConstraints, TrendDirection,
};

pub use executor::{MidCircuitDeviceExecutor, MidCircuitExecutor, ValidationResult};

pub use analytics::{
    anomaly::AnomalyDetector, causal::CausalAnalyzer, correlation::CorrelationAnalyzer,
    distribution::DistributionAnalyzer, statistical::StatisticalAnalyzer,
    time_series::TimeSeriesAnalyzer, AdvancedAnalyticsEngine,
};

pub use ml::{AdaptiveMeasurementManager, MLOptimizer, MeasurementPredictor};

pub use monitoring::{AlertThresholds, ExportFormat, OptimizationCache, PerformanceMonitor};

use crate::DeviceResult;
use std::time::Duration;

/// Convenience function to create a configured mid-circuit measurement executor
pub fn create_mid_circuit_executor(
    config: MidCircuitConfig,
    calibration_manager: crate::calibration::CalibrationManager,
) -> MidCircuitExecutor {
    MidCircuitExecutor::new(config, calibration_manager)
}

/// Convenience function to create default mid-circuit configuration
pub fn default_mid_circuit_config() -> MidCircuitConfig {
    MidCircuitConfig::default()
}

/// Convenience function to create high-performance mid-circuit configuration
pub fn high_performance_config() -> MidCircuitConfig {
    MidCircuitConfig {
        max_measurement_latency: 500.0, // 0.5ms
        enable_realtime_processing: true,
        measurement_buffer_size: 10000,
        classical_timeout: 100.0,
        enable_measurement_mitigation: true,
        enable_parallel_measurements: true,
        enable_adaptive_protocols: true,
        hardware_optimizations: HardwareOptimizations {
            batch_measurements: true,
            optimize_scheduling: true,
            use_native_protocols: true,
            measurement_compression: false,
            precompile_conditions: true,
        },
        validation_config: ValidationConfig {
            validate_capabilities: true,
            check_timing_constraints: true,
            validate_register_sizes: true,
            check_measurement_conflicts: true,
            validate_feedforward: true,
        },
        analytics_config: AdvancedAnalyticsConfig {
            enable_realtime_stats: true,
            enable_correlation_analysis: true,
            enable_time_series: true,
            enable_anomaly_detection: true,
            significance_threshold: 0.05,
            analysis_window_size: 1000,
            enable_distribution_fitting: true,
            enable_causal_inference: true,
        },
        ml_optimization_config: MLOptimizationConfig {
            enable_ml_optimization: true,
            model_types: vec![MLModelType::NeuralNetwork {
                hidden_layers: vec![64, 32],
            }],
            training_config: MLTrainingConfig::default(),
            feature_engineering: FeatureEngineeringConfig::default(),
            enable_transfer_learning: true,
            online_learning: OnlineLearningConfig::default(),
        },
        prediction_config: PredictionConfig {
            enable_prediction: true,
            prediction_horizon: 50,
            min_training_samples: 100,
            sequence_length: 10,
            time_series_config: TimeSeriesConfig::default(),
            uncertainty_config: UncertaintyConfig::default(),
            enable_ensemble: false,
        },
        adaptive_config: AdaptiveConfig {
            enable_adaptive_scheduling: true,
            enable_dynamic_thresholds: true,
            enable_protocol_adaptation: true,
            learning_rate: 0.001,
            baseline_update_rate: 0.1,
            drift_threshold: 0.1,
            adaptation_window: 50,
            enable_feedback_optimization: true,
            improvement_threshold: 0.05,
        },
    }
}

/// Convenience function to create analytics-focused configuration
pub fn analytics_focused_config() -> MidCircuitConfig {
    MidCircuitConfig {
        analytics_config: AdvancedAnalyticsConfig {
            enable_realtime_stats: true,
            enable_correlation_analysis: true,
            enable_time_series: true,
            enable_anomaly_detection: true,
            significance_threshold: 0.01,
            analysis_window_size: 5000,
            enable_distribution_fitting: true,
            enable_causal_inference: true,
        },
        ml_optimization_config: MLOptimizationConfig {
            enable_ml_optimization: true,
            model_types: vec![],
            training_config: Default::default(),
            feature_engineering: Default::default(),
            enable_transfer_learning: true,
            online_learning: Default::default(),
        },
        ..Default::default()
    }
}

/// Convenience function to create minimal configuration for basic measurements
pub fn minimal_config() -> MidCircuitConfig {
    MidCircuitConfig {
        max_measurement_latency: 2000.0, // 2ms
        enable_realtime_processing: false,
        measurement_buffer_size: 1000,
        classical_timeout: 50.0, // 50 microseconds
        enable_measurement_mitigation: false,
        enable_parallel_measurements: false,
        enable_adaptive_protocols: false,
        hardware_optimizations: HardwareOptimizations {
            batch_measurements: false,
            optimize_scheduling: false,
            use_native_protocols: false,
            measurement_compression: false,
            precompile_conditions: false,
        },
        validation_config: ValidationConfig {
            validate_capabilities: true,
            check_timing_constraints: false,
            validate_register_sizes: false,
            check_measurement_conflicts: false,
            validate_feedforward: false,
        },
        analytics_config: AdvancedAnalyticsConfig {
            enable_realtime_stats: true,
            enable_correlation_analysis: false,
            enable_time_series: false,
            enable_anomaly_detection: false,
            significance_threshold: 0.05,
            analysis_window_size: 100,
            enable_distribution_fitting: false,
            enable_causal_inference: false,
        },
        ml_optimization_config: MLOptimizationConfig {
            enable_ml_optimization: false,
            model_types: vec![],
            training_config: Default::default(),
            feature_engineering: Default::default(),
            enable_transfer_learning: false,
            online_learning: Default::default(),
        },
        prediction_config: PredictionConfig {
            enable_prediction: false,
            prediction_horizon: 5,
            min_training_samples: 100,
            sequence_length: 10,
            time_series_config: Default::default(),
            uncertainty_config: Default::default(),
            enable_ensemble: false,
        },
        adaptive_config: AdaptiveConfig {
            enable_adaptive_scheduling: false,
            enable_dynamic_thresholds: false,
            enable_protocol_adaptation: false,
            learning_rate: 0.01,
            baseline_update_rate: 0.1,
            drift_threshold: 0.1,
            adaptation_window: 100,
            enable_feedback_optimization: false,
            improvement_threshold: 0.05,
        },
    }
}

/// Initialize mid-circuit measurement system with custom configuration
pub fn initialize_system(config: MidCircuitConfig) -> DeviceResult<MidCircuitMeasurementSystem> {
    let calibration_manager = crate::calibration::CalibrationManager::new();
    let executor = MidCircuitExecutor::new(config.clone(), calibration_manager);
    let analytics_engine = analytics::AdvancedAnalyticsEngine::new(&config.analytics_config);
    let performance_monitor = monitoring::PerformanceMonitor::new();
    let optimization_cache = monitoring::OptimizationCache::new();

    Ok(MidCircuitMeasurementSystem {
        config,
        executor,
        analytics_engine,
        performance_monitor,
        optimization_cache,
    })
}

/// Complete mid-circuit measurement system
pub struct MidCircuitMeasurementSystem {
    pub config: MidCircuitConfig,
    pub executor: MidCircuitExecutor,
    pub analytics_engine: analytics::AdvancedAnalyticsEngine,
    pub performance_monitor: monitoring::PerformanceMonitor,
    pub optimization_cache: monitoring::OptimizationCache,
}

impl MidCircuitMeasurementSystem {
    /// Create new system with default configuration
    pub fn new() -> DeviceResult<Self> {
        initialize_system(default_mid_circuit_config())
    }

    /// Create new system with high-performance configuration
    pub fn new_high_performance() -> DeviceResult<Self> {
        initialize_system(high_performance_config())
    }

    /// Create new system with analytics-focused configuration
    pub fn new_analytics_focused() -> DeviceResult<Self> {
        initialize_system(analytics_focused_config())
    }

    /// Create new system with minimal configuration
    pub fn new_minimal() -> DeviceResult<Self> {
        initialize_system(minimal_config())
    }

    /// Execute a circuit with comprehensive measurement analysis
    pub async fn execute_and_analyze<const N: usize>(
        &mut self,
        circuit: &quantrs2_circuit::measurement::MeasurementCircuit<N>,
        device_executor: &dyn MidCircuitDeviceExecutor,
        shots: usize,
    ) -> DeviceResult<MidCircuitExecutionResult> {
        // Execute the circuit
        let mut result = self
            .executor
            .execute_circuit(circuit, device_executor, shots)
            .await?;

        // Record performance metrics
        self.performance_monitor
            .record_metrics(&result.performance_metrics)?;

        // Perform additional analytics if enabled
        if self.config.analytics_config.enable_realtime_stats {
            let enhanced_analytics = self
                .analytics_engine
                .analyze(&result.measurement_history, &result.execution_stats)
                .await?;
            result.analytics_results = enhanced_analytics;
        }

        Ok(result)
    }

    /// Get comprehensive system status
    pub fn get_system_status(&self) -> DeviceResult<SystemStatus> {
        let performance_summary = self.performance_monitor.get_performance_summary()?;
        let cache_stats = self.optimization_cache.get_cache_stats();
        let active_alerts = self.performance_monitor.check_alerts()?;

        let system_health = if active_alerts.iter().any(|a| a.severity == "Critical") {
            SystemHealth::Critical
        } else if !active_alerts.is_empty() {
            SystemHealth::Warning
        } else {
            SystemHealth::Healthy
        };

        Ok(SystemStatus {
            performance_summary,
            cache_stats: Some(cache_stats),
            active_alerts,
            system_health,
            uptime: Duration::from_secs(0), // Default uptime since overall_stats doesn't exist
        })
    }

    /// Export system metrics
    pub fn export_metrics(&self, format: monitoring::ExportFormat) -> DeviceResult<String> {
        self.performance_monitor.export_metrics(format)
    }

    /// Clear all caches and reset monitoring
    pub fn reset_system(&mut self) -> DeviceResult<()> {
        self.optimization_cache.clear();
        // Note: In a full implementation, we would also reset the performance monitor
        // and analytics engine state
        Ok(())
    }
}

/// System status information
#[derive(Debug, Clone)]
pub struct SystemStatus {
    pub performance_summary: monitoring::PerformanceSummary,
    pub cache_stats: Option<monitoring::CacheStatistics>,
    pub active_alerts: Vec<monitoring::PerformanceAlert>,
    pub system_health: SystemHealth,
    pub uptime: std::time::Duration,
}

/// System health status
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SystemHealth {
    Healthy,
    Warning,
    Critical,
}

impl Default for MidCircuitMeasurementSystem {
    fn default() -> Self {
        Self::new().expect("Failed to create default MidCircuitMeasurementSystem")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_creation() {
        let config = default_mid_circuit_config();
        assert!(config.max_measurement_latency > 0.0);
        assert!(config.measurement_buffer_size > 0);
    }

    #[test]
    fn test_high_performance_config() {
        let config = high_performance_config();
        assert!(config.enable_realtime_processing);
        assert!(config.hardware_optimizations.batch_measurements);
        assert!(config.analytics_config.enable_realtime_stats);
    }

    #[test]
    fn test_minimal_config() {
        let config = minimal_config();
        assert!(!config.enable_realtime_processing);
        assert!(!config.ml_optimization_config.enable_ml_optimization);
        assert!(!config.prediction_config.enable_prediction);
    }

    #[test]
    fn test_system_initialization() {
        let config = default_mid_circuit_config();
        let result = initialize_system(config);
        assert!(result.is_ok());
    }
}
