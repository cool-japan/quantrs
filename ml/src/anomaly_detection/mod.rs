//! Quantum Anomaly Detection Module
//!
//! This module implements quantum-enhanced anomaly detection algorithms that leverage
//! quantum computing principles for improved outlier detection, novelty detection,
//! and pattern recognition in both classical and quantum data.

pub mod algorithms;
pub mod config;
pub mod core;
pub mod metrics;
pub mod preprocessing;
pub mod streaming;

// Re-export main types for backward compatibility
pub use algorithms::*;
pub use config::*;
pub use core::*;
pub use metrics::*;
pub use preprocessing::*;
pub use streaming::*;
