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
//! let matrix = Array::<f64, _>::zeros((2, 2));
//! matrix[[0, 0]] = -1.0;  // Linear term for x
//! matrix[[1, 1]] = -1.0;  // Linear term for y
//! matrix[[0, 1]] = 2.0;   // Quadratic term for x*y
//! matrix[[1, 0]] = 2.0;   // Symmetric
//!
//! // Create variable map
//! let var_map = HashMap::new();
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
pub mod adaptive_optimization;
pub mod advanced_error_mitigation;
pub mod advanced_performance_analysis;
pub mod advanced_visualization;
pub mod ai_assisted_optimization;
pub mod analysis;
pub mod applications;
pub mod auto_array;
pub mod benchmark;
pub mod coherent_ising_machine;
pub mod compile;
pub mod constraints;
pub mod encoding;
pub mod gpu;
pub mod gpu_benchmark;
pub mod gpu_kernels;
pub mod gpu_memory_pool;
pub mod gpu_performance;
pub mod gpu_samplers;
pub mod hybrid_algorithms;
pub mod ml_guided_sampling;
pub mod optimization;
pub mod optimize;
pub mod parallel_tempering;
pub mod parallel_tempering_advanced;
pub mod performance_optimization;
pub mod performance_profiler;
pub mod problem_decomposition;
pub mod problem_dsl;
pub mod quantum_advantage_analysis;
pub mod quantum_annealing;
pub mod quantum_error_correction;
pub mod quantum_inspired_ml;
pub mod quantum_ml_integration;
pub mod quantum_neural_networks;
pub mod quantum_optimization_extensions;
pub mod quantum_state_tomography;
pub mod realtime_quantum_integration;
pub mod sampler;
pub mod sampler_framework;
pub mod scirs_stub;
pub mod sensitivity_analysis;
pub mod solution_clustering;
pub mod solution_debugger;
pub mod solution_statistics;
pub mod symbol;
pub mod tensor_network_sampler;
pub mod testing_framework;
pub mod topological_optimization;
pub mod variable_correlation;
pub mod variational_quantum_factoring;
pub mod visual_problem_builder;
pub mod visualization;

// Re-export key types for convenience
pub use advanced_error_mitigation::{
    create_advanced_error_mitigation_manager, create_lightweight_error_mitigation_manager,
    AdvancedErrorMitigationManager, ErrorMitigationConfig,
};
pub use advanced_performance_analysis::{
    create_comprehensive_analyzer, create_lightweight_analyzer, AdvancedPerformanceAnalyzer,
    AnalysisConfig,
};
pub use advanced_visualization::{
    create_advanced_visualization_manager, create_lightweight_visualization_manager,
    AdvancedVisualizationManager, VisualizationConfig,
};
pub use analysis::{calculate_diversity, cluster_solutions, visualize_energy_distribution};
#[cfg(feature = "dwave")]
pub use auto_array::AutoArray;
#[cfg(feature = "dwave")]
pub use compile::{Compile, PieckCompile};
#[cfg(feature = "gpu")]
pub use gpu::{gpu_solve_hobo, gpu_solve_qubo, is_available as is_gpu_available_internal};
pub use optimize::{calculate_energy, optimize_hobo, optimize_qubo};
pub use sampler::{ArminSampler, DWaveSampler, GASampler, MIKASAmpler, SASampler};
pub use scirs_stub::SCIRS2_AVAILABLE;
#[cfg(feature = "dwave")]
pub use symbol::{symbols, symbols_define, symbols_list, symbols_nbit};
pub use tensor_network_sampler::{
    create_mera_sampler, create_mps_sampler, create_peps_sampler, TensorNetworkSampler,
};
pub use visual_problem_builder::{
    BuilderConfig, ConstraintType, ExportFormat, ObjectiveExpression, VariableType, VisualProblem,
    VisualProblemBuilder,
};

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
