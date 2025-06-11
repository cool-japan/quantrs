//! Cloud Monitoring and Analytics Configuration
//!
//! This module provides comprehensive cloud monitoring capabilities for quantum computing
//! workloads, including performance monitoring, resource tracking, cost management,
//! security monitoring, alerting, and analytics.
//!
//! The module has been refactored into focused submodules for better organization:
//! - `performance`: Performance metrics and monitoring configuration
//! - `resource`: Resource monitoring and usage tracking  
//! - `cost`: Cost monitoring, budget tracking, and optimization
//! - `security`: Security monitoring and threat detection
//! - `alerting`: Alert configuration and notification systems
//! - `analytics`: Analytics, reporting, and anomaly detection
//! - `ml`: Machine learning and AutoML configurations

// Re-export everything from the new modular structure for backward compatibility
mod monitoring;
pub use monitoring::*;