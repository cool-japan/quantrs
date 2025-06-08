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
pub mod hobo;
pub mod ising;
pub mod partitioning;
pub mod qubo;
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
pub use dwave::{is_available as is_dwave_available, DWaveClient, DWaveError, DWaveResult};
pub use embedding::{Embedding, HardwareGraph, HardwareTopology, MinorMiner};
pub use hobo::{
    AuxiliaryVariable, ConstraintViolations, HigherOrderTerm, HoboAnalyzer, HoboProblem,
    HoboStats, QuboReduction, ReductionMethod, ReductionType,
};
pub use ising::{IsingError, IsingModel, IsingResult, QuboModel};
pub use partitioning::{
    BipartitionMethod, KernighanLinPartitioner, Partition, RecursiveBisectionPartitioner,
    SpectralPartitioner,
};
pub use qubo::{QuboBuilder, QuboError, QuboFormulation, QuboResult};
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
