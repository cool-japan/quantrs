//! Quantum Machine Learning Integration Module
//!
//! This module provides comprehensive quantum machine learning integration
//! with device management, training orchestration, and performance analytics.

// Type definitions (merged from split modules to avoid circular dependencies)
pub mod types;

// Function implementations (includes trait definitions)
pub mod functions;

// Trait implementations
mod alertmanager_traits;
mod frameworkbridge_traits;
mod hybridmloptimizer_traits;
mod qmldatapipeline_traits;
mod qmlintegrationconfig_traits;
mod qmlmonitoringconfig_traits;
mod qmloptimizationconfig_traits;
mod qmlresourceconfig_traits;
mod qmlresourcerequirements_traits;
mod qmltrainingconfig_traits;
mod simplemlanomalydetector_traits;
mod trainingperformancemonitor_traits;

// Re-export all types and functions
pub use functions::*;
pub use types::*;
