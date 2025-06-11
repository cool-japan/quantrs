//! Load balancing and traffic management configurations
//!
//! This module provides comprehensive load balancing configurations including:
//! - Load balancing strategies and health checks
//! - Traffic distribution and splitting
//! - Session management and affinity
//! - Canary deployment support

pub mod canary;
pub mod session;
pub mod strategies;
pub mod traffic;

// Re-export all types for convenience
pub use canary::*;
pub use session::*;
pub use strategies::*;
pub use traffic::*;

use serde::{Deserialize, Serialize};

/// Cloud load balancing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudLoadBalancingConfig {
    /// Load balancing strategies
    pub strategies: Vec<LoadBalancingStrategy>,
    /// Health checks
    pub health_checks: HealthCheckConfig,
    /// Traffic distribution
    pub traffic_distribution: TrafficDistributionConfig,
    /// Session management
    pub session_management: SessionManagementConfig,
}
