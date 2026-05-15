//! Cloud Orchestration and Load Balancing Configuration
//!
//! This module provides comprehensive configuration structures for cloud orchestration,
//! including performance optimization, load balancing, security, and budget management.
//!
//! The module is organized into focused sub-modules for better maintainability:
//! - `performance`: Performance optimization and QoS configurations
//! - `load_balancing`: Load balancing and traffic management
//! - `security`: Security, authentication, and compliance
//! - `budget`: Budget management and cost tracking

pub mod budget;
pub mod load_balancing;
pub mod performance;
pub mod security;

// Re-export all types for backward compatibility
pub use budget::*;
pub use load_balancing::*;
pub use performance::*;
pub use security::*;
