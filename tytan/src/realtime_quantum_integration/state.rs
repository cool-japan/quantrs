//! System state types for Real-time Quantum Computing Integration
//!
//! This module provides system state tracking types.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use super::hardware::DeviceStatus;
use super::types::{AlertType, ComponentStatus, IssueSeverity, SystemStatus};

/// System state tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    /// Overall system status
    pub overall_status: SystemStatus,
    /// Component states
    pub component_states: HashMap<String, ComponentState>,
    /// Active alerts
    pub active_alerts: Vec<ActiveAlert>,
    /// Performance summary
    pub performance_summary: PerformanceSummary,
    /// Resource utilization
    pub resource_utilization: SystemResourceUtilization,
    /// Last update timestamp
    pub last_update: SystemTime,
}

impl Default for SystemState {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemState {
    pub fn new() -> Self {
        Self {
            overall_status: SystemStatus::Healthy,
            component_states: HashMap::new(),
            active_alerts: vec![],
            performance_summary: PerformanceSummary {
                performance_score: 0.9,
                throughput: 100.0,
                latency_percentiles: HashMap::new(),
                error_rates: HashMap::new(),
                availability: 0.99,
            },
            resource_utilization: SystemResourceUtilization {
                cpu_utilization: 0.5,
                memory_utilization: 0.6,
                storage_utilization: 0.4,
                network_utilization: 0.3,
                quantum_utilization: Some(0.7),
            },
            last_update: SystemTime::now(),
        }
    }

    pub fn update_component_state(&mut self, component_id: &str, _status: &DeviceStatus) {
        let component_state = ComponentState {
            component_name: component_id.to_string(),
            status: ComponentStatus::Healthy, // Simplified
            last_heartbeat: SystemTime::now(),
            metrics: HashMap::new(),
            alerts: vec![],
        };

        self.component_states
            .insert(component_id.to_string(), component_state);
        self.last_update = SystemTime::now();
    }
}

/// Component state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentState {
    /// Component name
    pub component_name: String,
    /// Status
    pub status: ComponentStatus,
    /// Last heartbeat
    pub last_heartbeat: SystemTime,
    /// Metrics
    pub metrics: HashMap<String, f64>,
    /// Alerts
    pub alerts: Vec<String>,
}

/// Active alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveAlert {
    /// Alert ID
    pub alert_id: String,
    /// Alert type
    pub alert_type: AlertType,
    /// Severity
    pub severity: IssueSeverity,
    /// Message
    pub message: String,
    /// Timestamp
    pub timestamp: SystemTime,
    /// Acknowledged
    pub acknowledged: bool,
    /// Acknowledger
    pub acknowledged_by: Option<String>,
}

/// Performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    /// Overall performance score
    pub performance_score: f64,
    /// Throughput
    pub throughput: f64,
    /// Latency percentiles
    pub latency_percentiles: HashMap<String, Duration>,
    /// Error rates
    pub error_rates: HashMap<String, f64>,
    /// Availability percentage
    pub availability: f64,
}

/// System resource utilization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemResourceUtilization {
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Memory utilization
    pub memory_utilization: f64,
    /// Storage utilization
    pub storage_utilization: f64,
    /// Network utilization
    pub network_utilization: f64,
    /// Quantum resource utilization
    pub quantum_utilization: Option<f64>,
}
