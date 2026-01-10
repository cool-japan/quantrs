//! Hardware monitoring types for Real-time Quantum Computing Integration
//!
//! This module provides hardware monitor and device-related types.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant, SystemTime};

use super::config::MonitorConfig;
use super::metrics::{CalibrationData, DeviceMetrics};
use super::types::{
    AuthenticationType, ComponentStatus, ConnectionStatus, ConnectivityType, DeviceType,
    IssueSeverity, IssueType, MaintenanceType, MeasurementBasis, OverallStatus, WarningType,
};

/// Live hardware monitor for quantum devices
#[allow(dead_code)]
pub struct HardwareMonitor {
    /// Device information
    pub(crate) device_info: DeviceInfo,
    /// Current status
    pub(crate) current_status: DeviceStatus,
    /// Metrics history
    pub(crate) metrics_history: VecDeque<DeviceMetrics>,
    /// Calibration data
    pub(crate) calibration_data: CalibrationData,
    /// Monitor configuration
    pub(crate) monitor_config: MonitorConfig,
    /// Last update time
    pub(crate) last_update: Instant,
}

/// Device information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceInfo {
    /// Device ID
    pub device_id: String,
    /// Device type
    pub device_type: DeviceType,
    /// Device capabilities
    pub capabilities: DeviceCapabilities,
    /// Location information
    pub location: LocationInfo,
    /// Connection details
    pub connection: ConnectionInfo,
    /// Specifications
    pub specifications: DeviceSpecifications,
}

/// Device capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// Number of qubits
    pub num_qubits: usize,
    /// Supported gates
    pub supported_gates: Vec<String>,
    /// Connectivity graph
    pub connectivity: ConnectivityGraph,
    /// Maximum circuit depth
    pub max_circuit_depth: usize,
    /// Measurement capabilities
    pub measurement_capabilities: MeasurementCapabilities,
    /// Error rates
    pub error_rates: ErrorRates,
}

/// Connectivity graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectivityGraph {
    /// Adjacency matrix
    pub adjacency_matrix: Vec<Vec<bool>>,
    /// Connectivity type
    pub connectivity_type: ConnectivityType,
    /// Coupling strengths
    pub coupling_strengths: HashMap<(usize, usize), f64>,
}

/// Measurement capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementCapabilities {
    /// Measurement bases
    pub measurement_bases: Vec<MeasurementBasis>,
    /// Measurement fidelity
    pub measurement_fidelity: f64,
    /// Readout time
    pub readout_time: Duration,
    /// Simultaneous measurements
    pub simultaneous_measurements: bool,
}

/// Error rates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorRates {
    /// Single-qubit gate error
    pub single_qubit_gate_error: f64,
    /// Two-qubit gate error
    pub two_qubit_gate_error: f64,
    /// Measurement error
    pub measurement_error: f64,
    /// Decoherence rates
    pub decoherence_rates: DecoherenceRates,
}

/// Decoherence rates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoherenceRates {
    /// T1 time (relaxation)
    pub t1_time: Duration,
    /// T2 time (dephasing)
    pub t2_time: Duration,
    /// T2* time (inhomogeneous dephasing)
    pub t2_star_time: Duration,
}

/// Location information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocationInfo {
    /// Physical location
    pub physical_location: String,
    /// Timezone
    pub timezone: String,
    /// Coordinates
    pub coordinates: Option<(f64, f64)>,
    /// Network latency
    pub network_latency: Duration,
}

/// Connection information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionInfo {
    /// Endpoint URL
    pub endpoint: String,
    /// Authentication type
    pub auth_type: AuthenticationType,
    /// Connection status
    pub connection_status: ConnectionStatus,
    /// API version
    pub api_version: String,
    /// Rate limits
    pub rate_limits: RateLimits,
}

/// Rate limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimits {
    /// Requests per minute
    pub requests_per_minute: usize,
    /// Concurrent requests
    pub concurrent_requests: usize,
    /// Data transfer limits
    pub data_transfer_limits: DataTransferLimits,
}

/// Data transfer limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataTransferLimits {
    /// Maximum upload size
    pub max_upload_size: usize,
    /// Maximum download size
    pub max_download_size: usize,
    /// Bandwidth limit
    pub bandwidth_limit: usize,
}

/// Device specifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceSpecifications {
    /// Operating temperature
    pub operating_temperature: f64,
    /// Operating frequency range
    pub frequency_range: (f64, f64),
    /// Power consumption
    pub power_consumption: f64,
    /// Physical dimensions
    pub dimensions: PhysicalDimensions,
    /// Environmental requirements
    pub environmental_requirements: EnvironmentalRequirements,
}

/// Physical dimensions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalDimensions {
    /// Length in meters
    pub length: f64,
    /// Width in meters
    pub width: f64,
    /// Height in meters
    pub height: f64,
    /// Weight in kilograms
    pub weight: f64,
}

/// Environmental requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalRequirements {
    /// Temperature range
    pub temperature_range: (f64, f64),
    /// Humidity range
    pub humidity_range: (f64, f64),
    /// Vibration tolerance
    pub vibration_tolerance: f64,
    /// Electromagnetic shielding
    pub em_shielding_required: bool,
}

/// Device status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceStatus {
    /// Overall status
    pub overall_status: OverallStatus,
    /// Availability
    pub availability: Availability,
    /// Current load
    pub current_load: f64,
    /// Queue status
    pub queue_status: QueueStatus,
    /// Health indicators
    pub health_indicators: HealthIndicators,
    /// Last maintenance
    pub last_maintenance: SystemTime,
    /// Next scheduled maintenance
    pub next_maintenance: Option<SystemTime>,
}

impl Default for DeviceStatus {
    fn default() -> Self {
        Self {
            overall_status: OverallStatus::Online,
            availability: Availability {
                available: true,
                expected_available_time: None,
                availability_percentage: 0.99,
                planned_downtime: vec![],
            },
            current_load: 0.5,
            queue_status: QueueStatus {
                jobs_in_queue: 0,
                estimated_wait_time: Duration::ZERO,
                next_job_position: 0,
                processing_rate: 1.0,
            },
            health_indicators: HealthIndicators {
                system_temperature: 22.0,
                error_rate: 0.001,
                performance_metrics: PerformanceIndicators {
                    gate_fidelity: 0.99,
                    measurement_fidelity: 0.95,
                    coherence_times: DecoherenceRates {
                        t1_time: Duration::from_micros(100),
                        t2_time: Duration::from_micros(50),
                        t2_star_time: Duration::from_micros(30),
                    },
                    throughput: 1000.0,
                    latency: Duration::from_millis(10),
                },
                component_health: HashMap::new(),
                warning_flags: vec![],
            },
            last_maintenance: SystemTime::now(),
            next_maintenance: None,
        }
    }
}

/// Availability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Availability {
    /// Is device available
    pub available: bool,
    /// Expected availability time
    pub expected_available_time: Option<SystemTime>,
    /// Availability percentage
    pub availability_percentage: f64,
    /// Planned downtime
    pub planned_downtime: Vec<MaintenanceWindow>,
}

/// Maintenance window
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceWindow {
    /// Start time
    pub start_time: SystemTime,
    /// End time
    pub end_time: SystemTime,
    /// Maintenance type
    pub maintenance_type: MaintenanceType,
    /// Description
    pub description: String,
}

/// Queue status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueStatus {
    /// Number of jobs in queue
    pub jobs_in_queue: usize,
    /// Estimated wait time
    pub estimated_wait_time: Duration,
    /// Queue position for next job
    pub next_job_position: usize,
    /// Processing rate
    pub processing_rate: f64,
}

/// Health indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthIndicators {
    /// System temperature
    pub system_temperature: f64,
    /// Error rate
    pub error_rate: f64,
    /// Performance metrics
    pub performance_metrics: PerformanceIndicators,
    /// Component health
    pub component_health: HashMap<String, ComponentHealth>,
    /// Warning flags
    pub warning_flags: Vec<WarningFlag>,
}

/// Performance indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceIndicators {
    /// Gate fidelity
    pub gate_fidelity: f64,
    /// Measurement fidelity
    pub measurement_fidelity: f64,
    /// Coherence times
    pub coherence_times: DecoherenceRates,
    /// Throughput
    pub throughput: f64,
    /// Latency
    pub latency: Duration,
}

/// Component health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component name
    pub component_name: String,
    /// Health score
    pub health_score: f64,
    /// Status
    pub status: ComponentStatus,
    /// Last checked
    pub last_checked: SystemTime,
    /// Issues
    pub issues: Vec<ComponentIssue>,
}

/// Component issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentIssue {
    /// Issue type
    pub issue_type: IssueType,
    /// Severity
    pub severity: IssueSeverity,
    /// Description
    pub description: String,
    /// First occurrence
    pub first_occurrence: SystemTime,
    /// Frequency
    pub frequency: f64,
}

/// Warning flag
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarningFlag {
    /// Warning type
    pub warning_type: WarningType,
    /// Message
    pub message: String,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Acknowledged
    pub acknowledged: bool,
}

// Implementation for HardwareMonitor
impl HardwareMonitor {
    /// Create a new hardware monitor for the given device
    pub fn new(device_info: DeviceInfo) -> Self {
        Self {
            device_info,
            current_status: DeviceStatus::default(),
            metrics_history: VecDeque::new(),
            calibration_data: CalibrationData::default(),
            monitor_config: MonitorConfig::default(),
            last_update: Instant::now(),
        }
    }

    /// Update metrics for the device
    pub fn update_metrics(&mut self) -> Result<(), String> {
        use super::metrics::{EnvironmentalMetrics, HardwareMetrics, QuantumMetrics};

        // Simulate metric collection
        let metrics = DeviceMetrics {
            timestamp: SystemTime::now(),
            cpu_utilization: 0.5,
            memory_utilization: 0.6,
            network_utilization: 0.3,
            hardware_metrics: HardwareMetrics {
                temperatures: {
                    let mut temps = HashMap::new();
                    temps.insert("cpu".to_string(), 45.0);
                    temps.insert("quantum_chip".to_string(), 0.01);
                    temps
                },
                power_consumption: 150.0,
                vibration_levels: HashMap::new(),
                magnetic_fields: HashMap::new(),
            },
            quantum_metrics: QuantumMetrics {
                gate_fidelities: HashMap::new(),
                measurement_fidelities: HashMap::new(),
                coherence_measurements: HashMap::new(),
                crosstalk_matrix: None,
            },
            environmental_metrics: EnvironmentalMetrics {
                ambient_temperature: 22.0,
                humidity: 45.0,
                pressure: 1013.25,
                air_quality: None,
            },
        };

        self.metrics_history.push_back(metrics);
        if self.metrics_history.len() > 1000 {
            self.metrics_history.pop_front();
        }

        self.last_update = Instant::now();
        Ok(())
    }

    /// Get the current device status
    pub fn get_current_status(&self) -> DeviceStatus {
        self.current_status.clone()
    }
}
