//! High-level quantum annealing interface inspired by Tytan for the quantrs framework.
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
//! ```rust,no_run
//! use quantrs_tytan::{symbols, Compile, SASampler, Auto_array};
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
//! let result = solver.run(&qubo, 100).unwrap();
//!
//! // Display results
//! for r in &result {
//!     println!("{:?}", r);
//! }
//! ```

// Export modules
pub mod symbol;
pub mod compile;
pub mod sampler;
pub mod auto_array;
pub mod optimize;
pub mod gpu;
pub mod analysis;

// Re-export key types for convenience
#[cfg(feature = "dwave")]
pub use symbol::{symbols, symbols_list, symbols_define, symbols_nbit};
#[cfg(feature = "dwave")]
pub use compile::{Compile, PieckCompile};
pub use sampler::{SASampler, GASampler, ArminSampler, MIKASAmpler, DWaveSampler};
#[cfg(feature = "dwave")]
pub use auto_array::Auto_array;
pub use optimize::{optimize_qubo, optimize_hobo, calculate_energy};
#[cfg(feature = "gpu")]
pub use gpu::{gpu_solve_qubo, gpu_solve_hobo, is_available as is_gpu_available_internal};
pub use analysis::{cluster_solutions, calculate_diversity, visualize_energy_distribution};

// Expose Quantrs-anneal types as well for advanced usage
pub use quantrs_anneal::{IsingModel, QuboModel, IsingError, IsingResult};
pub use quantrs_anneal::{QuboBuilder, QuboError, QuboResult, QuboFormulation};

/// Check if the module is available
///
/// This function always returns `true` since the module
/// is available if you can import it.
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
            Some(platform) => {
                match ocl::Device::list_all(platform).unwrap_or_default().first() {
                    Some(_) => true,
                    None => false,
                }
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
pub fn is_gpu_available() -> bool {
    false
}

/// Print version information
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}