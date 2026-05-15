//! Quantum Algorithm Optimization Marketplace
//!
//! This module provides a comprehensive marketplace for quantum algorithm discovery, sharing,
//! optimization, and collaborative development. It includes algorithm registry, automated
//! optimization using SciRS2 analytics, performance benchmarking, collaborative features,
//! and integration with all quantum computing components for intelligent algorithm matching.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use quantrs2_circuit::prelude::*;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};

use serde::{Deserialize, Serialize};

// SciRS2 dependencies for advanced marketplace analytics and optimization
#[cfg(feature = "scirs2")]
use scirs2_graph::{
    betweenness_centrality, closeness_centrality, minimum_spanning_tree, dijkstra_path,
    strongly_connected_components, Graph,
};
#[cfg(feature = "scirs2")]
use scirs2_linalg::{det, eig, inv, matrix_norm, prelude::*, svd, LinalgError, LinalgResult};
#[cfg(feature = "scirs2")]
use scirs2_optimize::{minimize, OptimizeResult};
#[cfg(feature = "scirs2")]
use scirs2_stats::{corrcoef, distributions, mean, pearsonr, spearmanr, std, var};
use scirs2_stats::ttest::Alternative;

// Fallback implementations when SciRS2 is not available
#[cfg(not(feature = "scirs2"))]
mod fallback_scirs2;
#[cfg(not(feature = "scirs2"))]
use fallback_scirs2::*;

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::prelude::*;

use crate::{
    adaptive_compilation::AdaptiveCompilationConfig,
    backend_traits::{query_backend_capabilities, BackendCapabilities},
    calibration::{CalibrationManager, DeviceCalibration},
    integrated_device_manager::IntegratedQuantumDeviceManager,
    noise_model::CalibrationNoiseModel,
    topology::HardwareTopology,
    CircuitResult, DeviceError, DeviceResult,
};

// Module declarations
pub mod analytics;
pub mod benchmarking;
pub mod collaboration;
pub mod config;
pub mod discovery;
pub mod economic;
pub mod integration;
pub mod ml_integration;
pub mod optimization;
pub mod registry;
pub mod security;

// Re-exports for public API
pub use analytics::*;
pub use benchmarking::*;
pub use collaboration::*;
pub use config::*;
pub use discovery::*;
pub use economic::*;
pub use integration::*;
pub use ml_integration::*;
pub use optimization::*;
pub use registry::*;
pub use security::*;

#[cfg(not(feature = "scirs2"))]
pub use fallback_scirs2::*;

// Note: the full `QuantumAlgorithmMarketplace` struct and its implementation
// reside in the sibling `algorithm_marketplace` module
// (`device/src/algorithm_marketplace/mod.rs`), which predates this module.
// This module (`quantum_algorithm_marketplace`) provides the richer,
// sub-module-organized type hierarchy (analytics, benchmarking, collaboration,
// etc.) that was refactored out of the original monolithic file.  The two
// modules co-exist intentionally; no duplication needs to be resolved here.
