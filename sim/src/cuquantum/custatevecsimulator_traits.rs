//! # CuStateVecSimulator - Trait Implementations
//!
//! This module contains trait implementations for `CuStateVecSimulator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{CuQuantumConfig, CuStateVecSimulator, SimulationStats};

impl Default for CuStateVecSimulator {
    fn default() -> Self {
        Self::new(CuQuantumConfig::default()).unwrap_or_else(|_| Self {
            config: CuQuantumConfig::default(),
            device_info: None,
            stats: SimulationStats::default(),
            initialized: false,
            #[cfg(feature = "cuquantum")]
            handle: None,
            #[cfg(feature = "cuquantum")]
            state_buffer: None,
        })
    }
}
