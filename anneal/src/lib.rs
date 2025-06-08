//! Quantum annealing support for the QuantRS2 framework.
//!
//! This crate provides types and functions for quantum annealing,
//! including Ising model representation, QUBO problem formulation,
//! simulated quantum annealing, and D-Wave connectivity.
//!
//! # Features
//!
//! - Ising model representation with biases and couplings
//! - QUBO problem formulation with constraints
//! - Simulated quantum annealing using path integral Monte Carlo
//! - Classical simulated annealing using Metropolis algorithm
//! - D-Wave API client for connecting to quantum annealing hardware
//!
//! # Example
//!
//! ```rust
//! use quantrs2_anneal::{
//!     ising::IsingModel,
//!     simulator::{ClassicalAnnealingSimulator, AnnealingParams}
//! };
//!
//! // Create a simple 3-qubit Ising model
//! let mut model = IsingModel::new(3);
//! model.set_bias(0, 1.0).unwrap();
//! model.set_coupling(0, 1, -1.0).unwrap();
//!
//! // Configure annealing parameters
//! let mut params = AnnealingParams::new();
//! params.num_sweeps = 1000;
//! params.num_repetitions = 10;
//!
//! // Create an annealing simulator and solve the model
//! let simulator = ClassicalAnnealingSimulator::new(params).unwrap();
//! let result = simulator.solve(&model).unwrap();
//!
//! println!("Best energy: {}", result.best_energy);
//! println!("Best solution: {:?}", result.best_spins);
//! ```

// Export modules
pub mod chain_break;
pub mod compression;
pub mod dwave;
pub mod embedding;
pub mod flux_bias;
#[cfg(feature = "fujitsu")]
pub mod fujitsu;
pub mod hobo;
pub mod hybrid_solvers;
pub mod ising;
pub mod layout_embedding;
pub mod partitioning;
pub mod penalty_optimization;
pub mod problem_schedules;
pub mod qubo;
pub mod reverse_annealing;
pub mod simulator;

// Re-export key types for convenience
pub use chain_break::{
    ChainBreakResolver, ChainBreakStats, ChainStrengthOptimizer, HardwareSolution,
    LogicalProblem, ResolutionMethod, ResolvedSolution,
};
pub use compression::{
    BlockDetector, CompressedQubo, CompressionStats, CooCompressor, ReductionMapping,
    VariableReducer,
};
pub use dwave::{
    is_available as is_dwave_available, DWaveClient, DWaveError, DWaveResult, ProblemParams,
};
pub use embedding::{Embedding, HardwareGraph, HardwareTopology, MinorMiner};
pub use flux_bias::{
    FluxBiasOptimizer, FluxBiasConfig, FluxBiasResult, CalibrationData, MLFluxBiasOptimizer,
};
#[cfg(feature = "fujitsu")]
pub use fujitsu::{
    FujitsuClient, FujitsuError, FujitsuResult, FujitsuAnnealingParams, FujitsuHardwareSpec,
    GuidanceConfig, is_available as is_fujitsu_available,
};
pub use hobo::{
    AuxiliaryVariable, ConstraintViolations, HigherOrderTerm, HoboAnalyzer, HoboProblem,
    HoboStats, QuboReduction, ReductionMethod, ReductionType,
};
pub use hybrid_solvers::{
    HybridQuantumClassicalSolver, HybridSolverConfig, HybridSolverResult,
    VariationalHybridSolver,
};
pub use ising::{IsingError, IsingModel, IsingResult, QuboModel};
pub use layout_embedding::{
    LayoutAwareEmbedder, LayoutConfig, LayoutStats, MultiLevelEmbedder,
};
pub use partitioning::{
    BipartitionMethod, KernighanLinPartitioner, Partition, RecursiveBisectionPartitioner,
    SpectralPartitioner,
};
pub use penalty_optimization::{
    PenaltyOptimizer, PenaltyConfig, PenaltyStats, AdvancedPenaltyOptimizer,
    ConstraintPenaltyOptimizer, Constraint, ConstraintType,
};
pub use problem_schedules::{
    ProblemSpecificScheduler, ProblemType, ScheduleTemplate, AdaptiveScheduleOptimizer,
};
pub use qubo::{QuboBuilder, QuboError, QuboFormulation, QuboResult};
pub use reverse_annealing::{
    ReverseAnnealingSimulator, ReverseAnnealingParams, ReverseAnnealingSchedule,
    ReverseAnnealingScheduleBuilder,
};
pub use simulator::{
    AnnealingError, AnnealingParams, AnnealingResult, AnnealingSolution,
    ClassicalAnnealingSimulator, QuantumAnnealingSimulator, TemperatureSchedule,
    TransverseFieldSchedule,
};

/// Check if quantum annealing support is available
///
/// This function always returns `true` since the simulation capabilities
/// are always available.
pub fn is_available() -> bool {
    true
}

/// Check if hardware quantum annealing is available
///
/// This function checks if the D-Wave API client is available
/// and enabled via the "dwave" feature.
pub fn is_hardware_available() -> bool {
    dwave::is_available()
}
