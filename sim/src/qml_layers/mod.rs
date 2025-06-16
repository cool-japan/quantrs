//! Quantum Machine Learning Layers Framework - Refactored
//!
//! This module provides a comprehensive implementation of quantum machine learning layers,
//! refactored from the original quantum_machine_learning_layers.rs for better maintainability.

pub mod types;
pub mod layers;
pub mod training;
pub mod hardware;

// Re-export all public types and functions
pub use types::*;
pub use layers::*;
pub use training::*;
pub use hardware::*;

// Re-export the benchmark function
pub use crate::quantum_machine_learning_layers::benchmark_quantum_ml_layers;