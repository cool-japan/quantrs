//! Problem decomposition methods for large-scale optimization.
//!
//! This module provides various decomposition strategies including
//! graph partitioning, hierarchical solving, domain decomposition,
//! and constraint satisfaction problem decomposition.

pub mod graph_partitioner;
pub mod hierarchical_solver;
pub mod domain_decomposer;
pub mod csp_decomposer;
pub mod types;
pub mod utils;

// Re-export all public types
pub use graph_partitioner::*;
pub use hierarchical_solver::*;
pub use domain_decomposer::*;
pub use csp_decomposer::*;
pub use types::*;
pub use utils::*;