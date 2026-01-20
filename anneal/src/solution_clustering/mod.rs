//! Advanced Solution Clustering and Analysis for Quantum Annealing
//!
//! This module provides comprehensive clustering and analysis capabilities for quantum annealing
//! solutions, enabling deep insights into solution landscapes, convergence behavior, and
//! optimization performance. It includes multiple clustering algorithms, statistical analysis
//! tools, visualization support, and performance optimization suggestions.
//!
//! Key features:
//! - Multiple clustering algorithms (k-means, hierarchical, density-based, spectral)
//! - Solution landscape analysis and visualization
//! - Multi-modal solution detection and analysis
//! - Convergence analysis and trajectory clustering
//! - Statistical distribution analysis
//! - Solution quality assessment and ranking
//! - Performance optimization recommendations
//! - Parallel processing support for large solution sets

// Re-export public types and functions for backward compatibility
pub use algorithms::*;
pub use analyzer::*;
pub use config::*;
pub use error::*;
pub use types::*;
pub use utils::*;

// Module declarations
pub mod algorithms;
pub mod analyzer;
pub mod config;
pub mod error;
pub mod types;
pub mod utils;

#[cfg(test)]
pub mod tests;
