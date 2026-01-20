//! Metrics types for Real-time Quantum Computing Integration
//!
//! This module provides metrics and calibration-related types.

use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};

use super::hardware::DecoherenceRates;

/// Device metrics collected during monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceMetrics {
    /// Timestamp
    pub timestamp: SystemTime,
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Memory utilization
    pub memory_utilization: f64,
    /// Network utilization
    pub network_utilization: f64,
    /// Hardware metrics
    pub hardware_metrics: HardwareMetrics,
    /// Quantum metrics
    pub quantum_metrics: QuantumMetrics,
    /// Environmental metrics
    pub environmental_metrics: EnvironmentalMetrics,
}

/// Hardware-level metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareMetrics {
    /// Temperature readings
    pub temperatures: HashMap<String, f64>,
    /// Power consumption
    pub power_consumption: f64,
    /// Vibration levels
    pub vibration_levels: HashMap<String, f64>,
    /// Magnetic field measurements
    pub magnetic_fields: HashMap<String, f64>,
}

/// Quantum-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMetrics {
    /// Current gate fidelities
    pub gate_fidelities: HashMap<String, f64>,
    /// Current measurement fidelities
    pub measurement_fidelities: HashMap<usize, f64>,
    /// Coherence time measurements
    pub coherence_measurements: HashMap<usize, DecoherenceRates>,
    /// Cross-talk measurements
    pub crosstalk_matrix: Option<Array2<f64>>,
}

/// Environmental metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalMetrics {
    /// Ambient temperature
    pub ambient_temperature: f64,
    /// Humidity
    pub humidity: f64,
    /// Atmospheric pressure
    pub pressure: f64,
    /// Air quality index
    pub air_quality: Option<f64>,
}

/// Calibration data for quantum devices
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationData {
    /// Last calibration time
    pub last_calibration: SystemTime,
    /// Calibration results
    pub calibration_results: CalibrationResults,
    /// Calibration schedule
    pub calibration_schedule: CalibrationSchedule,
    /// Drift monitoring
    pub drift_monitoring: DriftMonitoring,
}

impl Default for CalibrationData {
    fn default() -> Self {
        Self {
            last_calibration: SystemTime::now(),
            calibration_results: CalibrationResults {
                gate_calibrations: HashMap::new(),
                measurement_calibrations: HashMap::new(),
                crosstalk_calibration: None,
                overall_score: 0.95,
            },
            calibration_schedule: CalibrationSchedule {
                regular_interval: Duration::from_secs(24 * 3600), // Daily
                next_calibration: SystemTime::now() + Duration::from_secs(24 * 3600),
                trigger_conditions: vec![],
                maintenance_integration: true,
            },
            drift_monitoring: DriftMonitoring {
                drift_parameters: HashMap::new(),
                prediction_model: None,
                drift_thresholds: HashMap::new(),
            },
        }
    }
}

/// Calibration results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResults {
    /// Gate calibrations
    pub gate_calibrations: HashMap<String, GateCalibration>,
    /// Measurement calibrations
    pub measurement_calibrations: HashMap<usize, MeasurementCalibration>,
    /// Crosstalk calibration
    pub crosstalk_calibration: Option<CrosstalkCalibration>,
    /// Overall calibration score
    pub overall_score: f64,
}

/// Gate calibration data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateCalibration {
    /// Gate name
    pub gate_name: String,
    /// Target qubits
    pub target_qubits: Vec<usize>,
    /// Fidelity achieved
    pub fidelity: f64,
    /// Parameters
    pub parameters: HashMap<String, f64>,
    /// Calibration time
    pub calibration_time: Duration,
}

/// Measurement calibration data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCalibration {
    /// Qubit index
    pub qubit_index: usize,
    /// Measurement fidelity
    pub fidelity: f64,
    /// Readout parameters
    pub readout_parameters: ReadoutParameters,
    /// Calibration matrices
    pub calibration_matrices: Option<Array2<f64>>,
}

/// Readout parameters for measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadoutParameters {
    /// Measurement pulse parameters
    pub pulse_parameters: HashMap<String, f64>,
    /// Integration weights
    pub integration_weights: Option<Array1<f64>>,
    /// Discrimination threshold
    pub discrimination_threshold: f64,
}

/// Crosstalk calibration data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrosstalkCalibration {
    /// Crosstalk matrix
    pub crosstalk_matrix: Array2<f64>,
    /// Mitigation strategy
    pub mitigation_strategy: CrosstalkMitigation,
    /// Effectiveness score
    pub effectiveness_score: f64,
}

/// Crosstalk mitigation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrosstalkMitigation {
    None,
    StaticCompensation,
    DynamicCompensation,
    PostProcessing,
}

/// Calibration schedule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSchedule {
    /// Regular calibration interval
    pub regular_interval: Duration,
    /// Next scheduled calibration
    pub next_calibration: SystemTime,
    /// Trigger conditions
    pub trigger_conditions: Vec<CalibrationTrigger>,
    /// Maintenance integration
    pub maintenance_integration: bool,
}

/// Calibration trigger conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CalibrationTrigger {
    TimeInterval(Duration),
    PerformanceDegradation(f64),
    EnvironmentalChange(f64),
    UserRequest,
    MaintenanceEvent,
}

/// Drift monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftMonitoring {
    /// Drift tracking parameters
    pub drift_parameters: HashMap<String, DriftParameter>,
    /// Drift prediction model
    pub prediction_model: Option<DriftPredictionModel>,
    /// Alert thresholds
    pub drift_thresholds: HashMap<String, f64>,
}

/// Drift parameter tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftParameter {
    /// Parameter name
    pub parameter_name: String,
    /// Current value
    pub current_value: f64,
    /// Baseline value
    pub baseline_value: f64,
    /// Drift rate
    pub drift_rate: f64,
    /// History
    pub value_history: VecDeque<(SystemTime, f64)>,
}

/// Drift prediction model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriftPredictionModel {
    /// Model type
    pub model_type: String,
    /// Model parameters
    pub parameters: HashMap<String, f64>,
    /// Prediction accuracy
    pub accuracy: f64,
    /// Last update
    pub last_update: SystemTime,
}
