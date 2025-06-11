//! Quantum Dimensionality Reduction Module
//!
//! This module implements quantum-enhanced dimensionality reduction algorithms including
//! linear methods, manifold learning, autoencoders, and feature selection techniques.

pub mod autoencoders;
pub mod config;
pub mod core;
pub mod feature_selection;
pub mod linear;
pub mod manifold;
pub mod metrics;
pub mod specialized;

// Re-export main types for backward compatibility
pub use autoencoders::*;
pub use config::*;
pub use core::*;
pub use feature_selection::*;
pub use linear::*;
pub use manifold::*;
pub use metrics::*;
pub use specialized::*;
