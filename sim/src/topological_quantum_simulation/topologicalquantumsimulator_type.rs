//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::AnyonModelImplementation;
use super::types::{
    BraidingOperation, SurfaceCode, TopologicalConfig, TopologicalLattice,
    TopologicalSimulationStats, TopologicalState,
};

/// Topological quantum simulator
pub struct TopologicalQuantumSimulator {
    /// Configuration
    pub(super) config: TopologicalConfig,
    /// Current topological state
    pub(super) state: TopologicalState,
    /// Lattice structure
    pub(super) lattice: TopologicalLattice,
    /// Anyon model implementation
    pub(super) anyon_model: Box<dyn AnyonModelImplementation + Send + Sync>,
    /// Error correction system
    pub(super) error_correction: Option<SurfaceCode>,
    /// Braiding history
    pub(super) braiding_history: Vec<BraidingOperation>,
    /// Simulation statistics
    pub(super) stats: TopologicalSimulationStats,
}
