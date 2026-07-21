//! Internal circuit-analysis types used by [`super::engine::CircuitMigrationEngine`]
//! while planning a migration (gate composition, connectivity, and resource
//! requirements relative to the target platform).

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use quantrs2_core::qubit::QubitId;

#[derive(Debug, Clone, Default)]
pub(crate) struct CircuitAnalysis {
    pub(crate) gate_analysis: GateAnalysis,
    pub(crate) connectivity_analysis: ConnectivityAnalysis,
    pub(crate) resource_analysis: ResourceAnalysis,
    pub(crate) compatibility_score: f64,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct GateAnalysis {
    pub(crate) gate_types: HashSet<String>,
    pub(crate) unsupported_gates: Vec<String>,
    pub(crate) decomposition_required: HashMap<String, usize>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ConnectivityAnalysis {
    pub(crate) required_connectivity: Vec<(QubitId, QubitId)>,
    pub(crate) connectivity_conflicts: Vec<(QubitId, QubitId)>,
    pub(crate) swap_overhead_estimate: usize,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct ResourceAnalysis {
    pub(crate) qubit_requirements: usize,
    pub(crate) memory_requirements: f64,
    pub(crate) execution_time_estimate: Duration,
}
