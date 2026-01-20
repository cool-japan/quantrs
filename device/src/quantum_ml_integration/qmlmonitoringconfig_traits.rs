//! # QMLMonitoringConfig - Trait Implementations
//!
//! This module contains trait implementations for `QMLMonitoringConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::time::Duration;

// Import types from sibling modules
use super::types::*;
// Merged into super::types
// Merged into super::types

impl Default for QMLMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_monitoring: true,
            collection_frequency: Duration::from_secs(30),
            performance_tracking: PerformanceTrackingConfig {
                track_training_metrics: true,
                track_inference_metrics: true,
                track_circuit_metrics: true,
                aggregation_window: Duration::from_secs(300),
                enable_trend_analysis: true,
            },
            resource_monitoring: ResourceMonitoringConfig {
                monitor_quantum_resources: true,
                monitor_classical_resources: true,
                monitor_memory: true,
                monitor_network: true,
                usage_thresholds: [
                    ("cpu".to_string(), 0.8),
                    ("memory".to_string(), 0.85),
                    ("quantum".to_string(), 0.9),
                ]
                .iter()
                .cloned()
                .collect(),
            },
            alert_config: AlertConfig {
                enabled: true,
                thresholds: [
                    ("error_rate".to_string(), 0.1),
                    ("resource_usage".to_string(), 0.9),
                    ("cost_spike".to_string(), 2.0),
                ]
                .iter()
                .cloned()
                .collect(),
                channels: vec![QMLAlertChannel::Log],
                escalation: AlertEscalation {
                    enabled: true,
                    levels: vec![
                        EscalationLevel {
                            name: "Warning".to_string(),
                            threshold_multiplier: 1.0,
                            channels: vec![QMLAlertChannel::Log],
                            actions: vec![EscalationAction::Notify],
                        },
                        EscalationLevel {
                            name: "Critical".to_string(),
                            threshold_multiplier: 2.0,
                            channels: vec![QMLAlertChannel::Log, QMLAlertChannel::Email],
                            actions: vec![EscalationAction::Notify, EscalationAction::Throttle],
                        },
                    ],
                    timeouts: [
                        ("warning".to_string(), Duration::from_secs(300)),
                        ("critical".to_string(), Duration::from_secs(60)),
                    ]
                    .iter()
                    .cloned()
                    .collect(),
                },
            },
        }
    }
}
