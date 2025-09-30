//! Comprehensive Device Performance Analytics Dashboard
//!
//! This module provides a comprehensive real-time performance analytics dashboard
//! that unifies monitoring, visualization, and intelligent insights across all quantum
//! device components using SciRS2's advanced analytics and machine learning capabilities.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use quantrs2_circuit::prelude::*;
use quantrs2_core::{
    error::{QuantRS2Error, QuantRS2Result},
    gate::GateOp,
    qubit::QubitId,
};

use serde::{Deserialize, Serialize};

// SciRS2 dependencies for advanced analytics
#[cfg(feature = "scirs2")]
use scirs2_graph::{
    betweenness_centrality, closeness_centrality, dijkstra_path, minimum_spanning_tree,
    strongly_connected_components, Graph,
};
#[cfg(feature = "scirs2")]
use scirs2_linalg::{det, eig, inv, matrix_norm, prelude::*, svd, LinalgError, LinalgResult};
#[cfg(feature = "scirs2")]
use scirs2_optimize::{minimize, OptimizeResult};
use scirs2_stats::ttest::Alternative;
#[cfg(feature = "scirs2")]
use scirs2_stats::{corrcoef, distributions, mean, pearsonr, spearmanr, std, var};

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
pub mod alerting;
pub mod config;
pub mod data_collection;
pub mod ml_analytics;
pub mod optimization;
pub mod reporting;
pub mod visualization;

// Re-exports for public API
pub use alerting::*;
pub use config::*;
pub use data_collection::*;
pub use ml_analytics::*;
pub use optimization::*;
pub use reporting::*;
pub use visualization::*;

#[cfg(not(feature = "scirs2"))]
pub use fallback_scirs2::*;

// TODO: Add implementation structs and functions that were in the original file
// This would include the PerformanceDashboard struct and its implementation
// For now, this refactoring focuses on organizing the configuration types
