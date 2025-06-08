//! High-level quantum annealing interface inspired by Tytan for the QuantRS2 framework.
//!
//! This crate provides a high-level interface for formulating and solving
//! quantum annealing problems, with support for multiple backend solvers.
//! It is inspired by the Python [Tytan](https://github.com/tytansdk/tytan) library.
//!
//! # Features
//!
//! - **Symbolic Problem Construction**: Define QUBO problems using symbolic expressions
//! - **Higher-Order Binary Optimization (HOBO)**: Support for terms beyond quadratic
//! - **Multiple Samplers**: Choose from various solvers
//! - **Auto Result Processing**: Automatically convert solutions to multi-dimensional arrays
//!
//! # Example
//!
//! Example with the `dwave` feature enabled:
//!
//! ```rust,no_run
//! # #[cfg(feature = "dwave")]
//! # fn dwave_example() {
//! use quantrs2_tytan::sampler::{SASampler, Sampler};
//! use quantrs2_tytan::symbol::symbols;
//! use quantrs2_tytan::compile::Compile;
//! use quantrs2_tytan::auto_array::Auto_array;
//!
//! // Define variables
//! let x = symbols("x");
//! let y = symbols("y");
//! let z = symbols("z");
//!
//! // Define expression (3 variables, want exactly 2 to be 1)
//! let h = (x + y + z - 2).pow(2);
//!
//! // Compile to QUBO
//! let (qubo, offset) = Compile::new(&h).get_qubo().unwrap();
//!
//! // Choose a sampler
//! let solver = SASampler::new(None);
//!
//! // Sample
//! let result = solver.run_qubo(&qubo, 100).unwrap();
//!
//! // Display results
//! for r in &result {
//!     println!("{:?}", r);
//! }
//! # }
//! ```
//!
//! Basic example without the `dwave` feature (no symbolic math):
//!
//! ```rust,no_run
//! use quantrs2_tytan::sampler::{SASampler, Sampler};
//! use std::collections::HashMap;
//! use ndarray::Array;
//!
//! // Create a simple QUBO matrix manually
//! let mut matrix = Array::<f64, _>::zeros((2, 2));
//! matrix[[0, 0]] = -1.0;  // Linear term for x
//! matrix[[1, 1]] = -1.0;  // Linear term for y
//! matrix[[0, 1]] = 2.0;   // Quadratic term for x*y
//! matrix[[1, 0]] = 2.0;   // Symmetric
//!
//! // Create variable map
//! let mut var_map = HashMap::new();
//! var_map.insert("x".to_string(), 0);
//! var_map.insert("y".to_string(), 1);
//!
//! // Choose a sampler
//! let solver = SASampler::new(None);
//!
//! // Sample by converting to the dynamic format for hobo
//! let matrix_dyn = matrix.into_dyn();
//! let result = solver.run_hobo(&(matrix_dyn, var_map), 100).unwrap();
//!
//! // Display results
//! for r in &result {
//!     println!("{:?}", r);
//! }
//! ```

// Export modules
pub mod analysis;
pub mod auto_array;
pub mod compile;
pub mod gpu;
pub mod optimize;
pub mod sampler;
pub mod scirs_stub;
pub mod symbol;

// Re-export key types for convenience
pub use analysis::{calculate_diversity, cluster_solutions, visualize_energy_distribution};
#[cfg(feature = "dwave")]
pub use auto_array::Auto_array;
#[cfg(feature = "dwave")]
pub use compile::{Compile, PieckCompile};
#[cfg(feature = "gpu")]
pub use gpu::{gpu_solve_hobo, gpu_solve_qubo, is_available as is_gpu_available_internal};
pub use optimize::{calculate_energy, optimize_hobo, optimize_qubo};
pub use sampler::{ArminSampler, DWaveSampler, GASampler, MIKASAmpler, SASampler};
pub use scirs_stub::SCIRS2_AVAILABLE;
#[cfg(feature = "dwave")]
pub use symbol::{symbols, symbols_define, symbols_list, symbols_nbit};

// Expose QuantRS2-anneal types as well for advanced usage
pub use quantrs2_anneal::{IsingError, IsingModel, IsingResult, QuboModel};
pub use quantrs2_anneal::{QuboBuilder, QuboError, QuboFormulation, QuboResult};

/// Check if the module is available
///
/// This function always returns `true` since the module
/// is available if you can import it.
#[must_use]
pub fn is_available() -> bool {
    true
}

/// Check if GPU acceleration is available
///
/// This function checks if GPU acceleration is available and enabled.
#[cfg(feature = "gpu")]
pub fn is_gpu_available() -> bool {
    #[cfg(feature = "ocl")]
    {
        // Try to get the first platform and device
        match ocl::Platform::list().first() {
            Some(platform) => match ocl::Device::list_all(platform).unwrap_or_default().first() {
                Some(_) => true,
                None => false,
            },
            None => false,
        }
    }

    #[cfg(not(feature = "ocl"))]
    {
        false
    }
}

#[cfg(not(feature = "gpu"))]
#[must_use]
pub fn is_gpu_available() -> bool {
    false
}

/// Print version information
#[must_use]
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
