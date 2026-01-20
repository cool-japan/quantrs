//! Monitoring and analytics for distributed systems

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

// Removed duplicate - using the one in types.rs

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MonitoringMetrics {
    pub system_health: f64,
    pub performance_score: f64,
    pub security_score: f64,
    pub availability_score: f64,
}
