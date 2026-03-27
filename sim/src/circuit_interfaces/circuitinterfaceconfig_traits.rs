//! # CircuitInterfaceConfig - Trait Implementations
//!
//! This module contains trait implementations for `CircuitInterfaceConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::CircuitInterfaceConfig;

impl Default for CircuitInterfaceConfig {
    fn default() -> Self {
        Self {
            auto_backend_selection: true,
            enable_optimization: true,
            max_statevector_qubits: 25,
            max_mps_bond_dim: 1024,
            parallel_compilation: true,
            enable_circuit_cache: true,
            max_cache_size: 10_000,
            enable_profiling: true,
        }
    }
}
