//! Real-Time Hardware Monitoring and Adaptive Compilation
//!
//! This module implements cutting-edge real-time monitoring of quantum annealing hardware
//! with intelligent adaptive compilation that optimizes problem mappings based on live
//! hardware performance data. It provides unprecedented control over quantum annealing
//! execution with millisecond-level adaptation to changing hardware conditions.
//!
//! Revolutionary Features:
//! - Real-time noise characterization and adaptation
//! - Dynamic qubit topology reconfiguration
//! - Adaptive chain strength optimization during execution
//! - Live error rate monitoring and mitigation
//! - Predictive hardware failure detection
//! - Quantum coherence preservation optimization
//! - Temperature-aware adaptive compilation
//! - Real-time calibration drift compensation

pub mod alerts;
pub mod collectors;
pub mod monitor_impl;
pub mod types;

// Re-export all public items
pub use alerts::*;
pub use collectors::*;
pub use monitor_impl::*;
pub use types::*;

use crate::applications::ApplicationResult;
use crate::HardwareTopology;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Create example real-time hardware monitor
pub fn create_example_hardware_monitor() -> ApplicationResult<RealTimeHardwareMonitor> {
    let config = MonitoringConfig::default();
    let monitor = RealTimeHardwareMonitor::new(config);

    // Register example device
    let device = MonitoredDevice {
        device_id: "dwave_advantage_4_1".to_string(),
        device_info: DeviceInfo {
            name: "D-Wave Advantage 4.1".to_string(),
            num_qubits: 5000,
            max_connectivity: 15,
            supported_operations: vec![
                QuantumOperation::IsingAnnealing,
                QuantumOperation::QUBOOptimization,
                QuantumOperation::ReverseAnnealing,
            ],
            temperature_range: (0.01, 0.02),
            coherence_characteristics: CoherenceCharacteristics {
                t1_relaxation: Duration::from_micros(100),
                t2_dephasing: Duration::from_micros(50),
                coherence_factor: 0.95,
                decoherence_sources: vec![
                    DecoherenceSource::ThermalNoise,
                    DecoherenceSource::FluxNoise,
                ],
            },
        },
        status: DeviceStatus::Online,
        performance_metrics: Arc::new(RwLock::new(DevicePerformanceMetrics {
            error_rate: 0.02,
            temperature: 0.015,
            coherence_time: Duration::from_micros(80),
            noise_level: 0.05,
            success_rate: 0.95,
            execution_speed: 1.5,
            queue_depth: 2,
            last_update: Instant::now(),
            performance_trend: PerformanceTrend {
                error_rate_trend: TrendDirection::Stable,
                temperature_trend: TrendDirection::Stable,
                coherence_trend: TrendDirection::Improving,
                overall_trend: TrendDirection::Stable,
                confidence: 0.8,
            },
        })),
        topology: HardwareTopology::Pegasus(16),
        connection: DeviceConnection::Custom("dwave_cloud".to_string()),
        monitoring_history: Arc::new(Mutex::new(std::collections::VecDeque::new())),
        noise_profile: Arc::new(RwLock::new(NoiseProfile {
            qubit_noise: vec![0.01; 5000],
            coupling_noise: vec![vec![0.005; 5000]; 5000],
            temporal_noise: TemporalNoiseProfile {
                autocorrelation: vec![1.0, 0.8, 0.6, 0.4, 0.2],
                correlation_times: vec![Duration::from_micros(10), Duration::from_micros(50)],
                memory_effects: vec![0.1, 0.05, 0.02],
                burst_patterns: vec![],
            },
            spectral_noise: SpectralNoiseProfile {
                power_spectrum: vec![1.0; 100],
                frequency_bins: (0..100).map(|i| f64::from(i) * 0.1).collect(),
                dominant_frequencies: vec![1.0, 2.5, 5.0],
                flicker_noise_params: FlickerNoiseParams {
                    amplitude: 0.01,
                    exponent: 1.0,
                    corner_frequency: 1.0,
                },
            },
            noise_correlations: NoiseCorrelationMatrix {
                spatial_correlations: vec![vec![0.0; 5000]; 5000],
                temporal_correlations: vec![0.5, 0.3, 0.1],
                cross_correlations: HashMap::new(),
            },
            last_update: Instant::now(),
        })),
        calibration_data: Arc::new(RwLock::new(CalibrationData {
            bias_calibration: vec![1.0; 5000],
            coupling_calibration: vec![vec![1.0; 5000]; 5000],
            schedule_calibration: ScheduleCalibration {
                optimal_anneal_time: Duration::from_micros(20),
                shape_parameters: vec![1.0, 0.5, 0.2],
                pause_points: vec![0.3, 0.7],
                ramp_rates: vec![0.1, 0.05],
            },
            temperature_calibration: TemperatureCalibration {
                offset_correction: 0.001,
                scaling_factor: 1.0,
                stability_map: vec![vec![1.0; 100]; 100],
            },
            last_calibration: Instant::now(),
            calibration_validity: Duration::from_secs(3600),
        })),
    };

    monitor.register_device(device)?;

    Ok(monitor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn test_monitor_creation() {
        let config = MonitoringConfig::default();
        let monitor = RealTimeHardwareMonitor::new(config);

        assert!(!monitor.monitoring_active.load(Ordering::Relaxed));
    }

    #[test]
    fn test_device_registration() {
        let monitor =
            create_example_hardware_monitor().expect("should create example hardware monitor");

        let devices = monitor
            .devices
            .read()
            .expect("should acquire read lock on devices");
        assert_eq!(devices.len(), 1);
        assert!(devices.contains_key("dwave_advantage_4_1"));
    }

    #[test]
    fn test_metrics_collection_config() {
        let config = MetricsCollectionConfig::default();
        assert!(config.enabled_metrics.contains(&MetricType::ErrorRate));
        assert!(config.enabled_metrics.contains(&MetricType::Temperature));
    }

    #[test]
    fn test_adaptive_compiler() {
        let compiler = AdaptiveCompiler::new();
        assert!(compiler.config.enable_realtime_recompilation);
        assert_eq!(compiler.config.cache_size, 1000);
    }

    #[test]
    fn test_alert_system() {
        let alert_system = AlertSystem::new();
        assert_eq!(alert_system.config.max_active_alerts, 100);
        assert!(alert_system.active_alerts.is_empty());
    }

    #[test]
    fn test_failure_detector() {
        let detector = PredictiveFailureDetector::new();
        assert_eq!(detector.config.confidence_threshold, 0.8);
        assert!(detector.models.is_empty());
    }

    #[test]
    fn test_performance_optimizer() {
        let optimizer = RealTimePerformanceOptimizer::new();
        assert_eq!(optimizer.config.improvement_threshold, 0.05);
        assert_eq!(optimizer.config.max_concurrent_optimizations, 3);
    }
}
