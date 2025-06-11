//! Load balancing strategies and health check configurations

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Load balancing strategies
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    WeightedRoundRobin,
    LeastConnections,
    LeastResponseTime,
    IPHash,
    GeographicProximity,
    Custom(String),
}

/// Health check configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Check interval
    pub interval: Duration,
    /// Timeout
    pub timeout: Duration,
    /// Unhealthy threshold
    pub unhealthy_threshold: usize,
    /// Healthy threshold
    pub healthy_threshold: usize,
    /// Check types
    pub check_types: Vec<HealthCheckType>,
}

/// Health check types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum HealthCheckType {
    HTTP,
    HTTPS,
    TCP,
    UDP,
    Custom(String),
}
