//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "optimize")]
use crate::optirs_integration::{OptiRSConfig, OptiRSQuantumOptimizer};
use scirs2_core::random::prelude::*;
use std::sync::{Arc, Mutex};

use super::types::{ParameterDatabase, QAOAConfig, QAOAGraph, QAOAProblemType, QAOAStats};

/// Main QAOA optimizer
pub struct QAOAOptimizer {
    /// Configuration
    pub(super) config: QAOAConfig,
    /// Problem graph
    pub(super) graph: QAOAGraph,
    /// Problem type
    pub(super) problem_type: QAOAProblemType,
    /// Current parameters
    pub(super) gammas: Vec<f64>,
    pub(super) betas: Vec<f64>,
    /// Best parameters found
    pub(super) best_gammas: Vec<f64>,
    pub(super) best_betas: Vec<f64>,
    /// Best cost found
    pub(super) best_cost: f64,
    /// Classical optimal solution (if known)
    pub(super) classical_optimum: Option<f64>,
    /// Optimization statistics
    pub(super) stats: QAOAStats,
    /// Parameter transfer database
    pub(super) parameter_database: Arc<Mutex<ParameterDatabase>>,
    /// `OptiRS` optimizer (optional, for `OptiRS` strategy)
    #[cfg(feature = "optimize")]
    pub(super) optirs_optimizer: Option<OptiRSQuantumOptimizer>,
}
