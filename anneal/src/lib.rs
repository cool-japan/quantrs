//! Quantum annealing support for the quantrs framework.
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
//! use quantrs_anneal::{
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
pub mod ising;
pub mod qubo;
pub mod simulator;
pub mod dwave;

// Re-export key types for convenience
pub use ising::{IsingModel, QuboModel, IsingError, IsingResult};
pub use qubo::{QuboBuilder, QuboError, QuboResult, QuboFormulation};
pub use simulator::{
    AnnealingParams, AnnealingError, AnnealingResult, AnnealingSolution,
    ClassicalAnnealingSimulator, QuantumAnnealingSimulator,
    TransverseFieldSchedule, TemperatureSchedule
};
pub use dwave::{DWaveClient, DWaveError, DWaveResult, is_available as is_dwave_available};

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