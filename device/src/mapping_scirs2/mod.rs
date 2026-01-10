//! Advanced qubit mapping using SciRS2 graph algorithms
//!
//! This module provides state-of-the-art qubit mapping and routing algorithms
//! leveraging SciRS2's comprehensive graph analysis capabilities.

// Re-export all public types
pub use analytics::*;
pub use config::*;
pub use core::*;
pub use graph_analysis::*;
pub use mapping_algorithms::*;
pub use ml_integration::*;
pub use optimization::*;
pub use types::*;
pub use utils::*;

// Module declarations
pub mod analytics;
pub mod config;
pub mod core;
pub mod graph_analysis;
pub mod mapping_algorithms;
pub mod ml_integration;
pub mod optimization;
pub mod types;
pub mod utils;

// Common imports for all submodules
pub use std::cmp::Ordering;
pub use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet, VecDeque};
pub use std::sync::{Arc, Mutex, RwLock};
pub use std::time::{Duration, Instant, SystemTime};

pub use scirs2_core::random::prelude::*;
pub use serde::{Deserialize, Serialize};
#[cfg(feature = "scheduling")]
pub use tokio::sync::{Mutex as AsyncMutex, RwLock as AsyncRwLock};

// Import only what we need from quantrs2_circuit to avoid name conflicts
pub use quantrs2_circuit::builder::Circuit;

pub use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};

#[cfg(feature = "scirs2")]
pub use scirs2_graph::{
    astar_search, astar_search_digraph, barabasi_albert_graph, betweenness_centrality,
    closeness_centrality, clustering_coefficient, diameter, dijkstra_path, eigenvector_centrality,
    erdos_renyi_graph, graph_density, k_core_decomposition, louvain_communities_result,
    maximum_bipartite_matching, minimum_cut, minimum_spanning_tree, pagerank, radius,
    spectral_radius, strongly_connected_components, topological_sort, watts_strogatz_graph,
    DiGraph, Edge, Graph, GraphError, Node, Result as GraphResult,
};
// Import specific functions from scirs2_linalg to avoid CalibrationMethod ambiguity
#[cfg(feature = "scirs2")]
pub use scirs2_linalg::{eig, matrix_norm, svd, LinalgResult};
#[cfg(feature = "scirs2")]
pub use scirs2_optimize::{minimize, OptimizeResult};
#[cfg(feature = "scirs2")]
pub use scirs2_stats::{corrcoef, mean, pearsonr, std};

pub use petgraph::graph::{NodeIndex, UnGraph};
pub use petgraph::Graph as PetGraph;
pub use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};

pub use crate::{
    calibration::DeviceCalibration,
    routing_advanced::{AdvancedRoutingResult, RoutingMetrics, SwapOperation},
    topology::HardwareTopology,
    DeviceError, DeviceResult,
};
