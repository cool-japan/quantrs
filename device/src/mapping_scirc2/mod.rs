//! Advanced qubit mapping using SciRS2 graph algorithms
//!
//! This module provides state-of-the-art qubit mapping and routing algorithms
//! leveraging SciRS2's comprehensive graph analysis capabilities.

// Re-export all public types
pub use config::*;
pub use types::*;
pub use core::*;
pub use graph_analysis::*;
pub use mapping_algorithms::*;
pub use optimization::*;
pub use ml_integration::*;
pub use analytics::*;
pub use utils::*;

// Module declarations
pub mod config;
pub mod types;
pub mod core;
pub mod graph_analysis;
pub mod mapping_algorithms;
pub mod optimization;
pub mod ml_integration;
pub mod analytics;
pub mod utils;

// Common imports for all submodules
pub use std::cmp::Ordering;
pub use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque, BTreeMap};
pub use std::sync::{Arc, RwLock, Mutex};
pub use std::time::{Duration, Instant, SystemTime};

pub use serde::{Deserialize, Serialize};
#[cfg(feature = "scheduling")]
pub use tokio::sync::{Mutex as AsyncMutex, RwLock as AsyncRwLock};
pub use rand::prelude::*;

pub use quantrs2_circuit::prelude::*;
pub use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};

#[cfg(feature = "scirs2")]
pub use scirs2_graph::{
    astar_search, astar_search_digraph, barabasi_albert_graph, betweenness_centrality,
    closeness_centrality, clustering_coefficient, diameter, eigenvector_centrality,
    erdos_renyi_graph, graph_density, k_core_decomposition, louvain_communities_result,
    maximum_bipartite_matching, minimum_cut, minimum_spanning_tree, pagerank, radius,
    dijkstra_path, shortest_path_digraph, spectral_radius, strongly_connected_components,
    topological_sort, watts_strogatz_graph, DiGraph, Edge, Graph, GraphError, Node,
    Result as GraphResult,
};
#[cfg(feature = "scirs2")]
pub use scirs2_linalg::{eig, matrix_norm, prelude::*, svd, LinalgResult};
#[cfg(feature = "scirs2")]
pub use scirs2_optimize::{minimize, OptimizeResult};
#[cfg(feature = "scirs2")]
pub use scirs2_stats::{corrcoef, mean, pearsonr, std};

pub use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
pub use petgraph::graph::{NodeIndex, UnGraph};
pub use petgraph::Graph as PetGraph;

pub use crate::{
    calibration::DeviceCalibration,
    routing_advanced::{AdvancedRoutingResult, RoutingMetrics, SwapOperation},
    topology::HardwareTopology,
    DeviceError, DeviceResult,
};