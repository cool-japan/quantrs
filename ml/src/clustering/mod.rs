//! Quantum Clustering Module
//!
//! This module implements quantum-enhanced clustering algorithms including
//! centroid-based, density-based, model-based, and hierarchical clustering methods.

pub mod centroid_based;
pub mod config;
pub mod core;
pub mod density_based;
pub mod hierarchy_based;
pub mod metrics;
pub mod model_based;
pub mod specialized;

// Re-export main types for backward compatibility
pub use centroid_based::*;
pub use config::*;
pub use core::ClusteringMetrics as CoreClusteringMetrics;
pub use core::{
    create_default_quantum_dbscan, create_default_quantum_kmeans, ClusteringResult,
    QuantumClusterer,
};
pub use density_based::*;
pub use hierarchy_based::*;
pub use metrics::*;
pub use model_based::*;
pub use specialized::*;
