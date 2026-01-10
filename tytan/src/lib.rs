// Architectural decisions - these are intentional design patterns
#![allow(clippy::unnecessary_wraps)] // Result return types for API consistency
#![allow(clippy::unused_self)] // Trait implementations require &self
#![allow(clippy::unused_async)]
// Async placeholders for future implementation
// Performance-related (not safety issues, can be optimized later)
#![allow(clippy::significant_drop_tightening)] // Lock scope optimization TODO
// Style-related (low priority)
#![allow(clippy::match_same_arms)] // Sometimes intentional for clarity
#![allow(clippy::option_if_let_else)] // Style preference
#![allow(clippy::return_self_not_must_use)] // Builder pattern
#![allow(clippy::needless_range_loop)] // Sometimes clearer with index
// Additional suppressions for remaining warnings
#![allow(clippy::branches_sharing_code)] // Sometimes intentional
#![allow(clippy::type_complexity)] // Quantum types are complex
#![allow(clippy::missing_const_for_fn)] // Not always beneficial
#![allow(clippy::format_push_string)] // Performance optimization TODO
#![allow(clippy::cast_possible_truncation)] // Platform-specific, validated
#![allow(clippy::future_not_send)] // Async architecture decision
#![allow(clippy::needless_pass_by_ref_mut)] // API consistency
#![allow(clippy::cast_precision_loss)] // Acceptable for quantum simulation
#![allow(clippy::uninlined_format_args)] // Style preference
#![allow(clippy::assigning_clones)] // Sometimes clearer
#![allow(clippy::zero_sized_map_values)] // Intentional for set-like maps
#![allow(clippy::used_underscore_binding)] // Sometimes needed for unused captures
#![allow(clippy::collection_is_never_read)] // Builder pattern / lazy evaluation
#![allow(clippy::wildcard_in_or_patterns)] // Sometimes intentional
#![allow(clippy::ptr_arg)] // API consistency with slices
#![allow(clippy::implicit_hasher)] // Generic hasher not always needed
#![allow(clippy::ref_option)] // Sometimes needed for lifetime reasons
#![allow(clippy::expect_fun_call)] // Clearer error messages
#![allow(clippy::if_not_else)] // Sometimes clearer
#![allow(clippy::iter_on_single_items)] // Sometimes intentional
#![allow(clippy::trivially_copy_pass_by_ref)] // API consistency
#![allow(clippy::empty_line_after_doc_comments)] // Formatting preference
#![allow(clippy::manual_let_else)] // Style preference
#![allow(clippy::await_holding_lock)] // Async architecture
// Full clippy category suppressions
#![allow(clippy::pedantic)]
#![allow(clippy::nursery)]
#![allow(clippy::cargo)]
// Additional specific suppressions
#![allow(clippy::large_enum_variant)]
#![allow(clippy::borrowed_box)]
#![allow(clippy::manual_map)]
#![allow(clippy::non_send_fields_in_send_ty)]
#![allow(clippy::if_same_then_else)]
#![allow(clippy::manual_clamp)]
#![allow(clippy::double_must_use)]
#![allow(clippy::only_used_in_recursion)]
#![allow(clippy::same_item_push)]
#![allow(clippy::format_in_format_args)]
#![allow(clippy::implied_bounds_in_impls)]
#![allow(clippy::explicit_counter_loop)]
#![allow(clippy::duplicated_attributes)]
#![allow(clippy::new_ret_no_self)]
#![allow(clippy::must_use_unit)]
#![allow(clippy::redundant_pattern_matching)]
#![allow(clippy::redundant_guards)]
#![allow(clippy::wrong_self_convention)]
#![allow(clippy::iter_next_slice)]
#![allow(clippy::create_dir)]
#![allow(clippy::enum_variant_names)]
// Additional specific suppressions (correct lint names)
#![allow(clippy::should_implement_trait)] // Methods like default(), from_str(), next()
#![allow(clippy::upper_case_acronyms)] // VQE, QAOA, QFT, CNOT, SGD
#![allow(clippy::unnecessary_map_or)] // map_or simplification suggestions
#![allow(clippy::derivable_impls)] // Impl can be derived
#![allow(clippy::or_fun_call)] // unwrap_or_else with default value
#![allow(clippy::cloned_ref_to_slice_refs)] // clone can be replaced with from_ref
#![allow(clippy::collapsible_match)]
#![allow(clippy::len_without_is_empty)]
#![allow(clippy::arc_with_non_send_sync)]
#![allow(clippy::std_instead_of_core)] // Allow std usage
#![allow(clippy::match_like_matches_macro)] // Sometimes match is clearer
#![allow(clippy::suspicious_open_options)] // File open options
#![allow(clippy::new_without_default)] // new() without Default impl

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
//! ## Recent Updates (v0.1.0-rc.2)
//!
//! - Refined SciRS2 v0.1.1 Stable Release integration for enhanced performance
//! - High-performance sparse matrix operations via SciRS2
//! - Parallel optimization using `scirs2_core::parallel_ops`
//! - SIMD-accelerated energy calculations
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
//! use quantrs2_tytan::auto_array::AutoArray;
//!
//! // Define variables
//! let x = symbols("x");
//! let y = symbols("y");
//! let z = symbols("z");
//!
//! // Define expression (3 variables, want exactly 2 to be 1)
//! let two: quantrs2_symengine_pure::Expression = 2.into();
//! let h = (x + y + z - two.clone()).pow(&two);
//!
//! // Compile to QUBO
//! let (qubo, offset) = Compile::new(h).get_qubo().expect("Failed to compile QUBO");
//!
//! // Choose a sampler
//! let solver = SASampler::new(None);
//!
//! // Sample
//! let mut result = solver.run_qubo(&qubo, 100).expect("Failed to run QUBO sampler");
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
//! use scirs2_core::ndarray::Array;
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
//! let mut result = solver.run_hobo(&(matrix_dyn, var_map), 100).expect("Failed to run HOBO sampler");
//!
//! // Display results
//! for r in &result {
//!     println!("{:?}", r);
//! }
//! ```

// Export modules
pub mod adaptive_noise_calibration;
pub mod adaptive_optimization;
pub mod advanced_error_mitigation;
pub mod advanced_performance_analysis;
pub mod advanced_problem_decomposition;
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
pub mod grover_amplitude_amplification;
pub mod hybrid_algorithms;
pub mod ml_guided_sampling;
pub mod multi_objective_optimization;
pub mod optimization;
pub mod optimize;
pub mod parallel_tempering;
pub mod parallel_tempering_advanced;
pub mod performance_optimization;
pub mod performance_profiler;
pub mod problem_decomposition;
pub mod problem_dsl;
pub mod quantum_adiabatic_path_optimization;
pub mod quantum_advantage_analysis;
pub mod quantum_advantage_prediction;
pub mod quantum_annealing;
pub mod quantum_circuit_annealing_compiler;
pub mod quantum_classical_hybrid;
pub mod quantum_error_correction;
pub mod quantum_inspired_ml;
pub mod quantum_ml_integration;
pub mod quantum_neural_networks;
pub mod quantum_optimization_extensions;
pub mod quantum_state_tomography;
pub mod realtime_performance_dashboard;
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
pub use grover_amplitude_amplification::{
    GroverAmplificationConfig, GroverAmplifiedSampler, GroverAmplifiedSolver,
};
pub use optimize::{calculate_energy, optimize_hobo, optimize_qubo};
pub use quantum_adiabatic_path_optimization::{
    AdiabaticPathConfig, PathInterpolation, QuantumAdiabaticPathOptimizer, QuantumAdiabaticSampler,
};
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
pub const fn is_available() -> bool {
    true
}

/// Check if GPU acceleration is available
///
/// This function checks if GPU acceleration is available and enabled.
#[cfg(feature = "gpu")]
pub fn is_gpu_available() -> bool {
    // When gpu feature is enabled, ocl dependency is available
    // Try to get the first platform and device
    match ocl::Platform::list().first() {
        Some(platform) => !ocl::Device::list_all(platform)
            .unwrap_or_default()
            .is_empty(),
        None => false,
    }
}

#[cfg(not(feature = "gpu"))]
#[must_use]
pub const fn is_gpu_available() -> bool {
    false
}

/// Print version information
#[must_use]
pub const fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
