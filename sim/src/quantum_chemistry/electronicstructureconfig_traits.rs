//! # ElectronicStructureConfig - Trait Implementations
//!
//! This module contains trait implementations for `ElectronicStructureConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{
    ElectronicStructureConfig, ElectronicStructureMethod, FermionMapping, VQEConfig,
};

impl Default for ElectronicStructureConfig {
    fn default() -> Self {
        Self {
            method: ElectronicStructureMethod::VQE,
            convergence_threshold: 1e-6,
            max_scf_iterations: 100,
            active_space: None,
            enable_second_quantization_optimization: true,
            fermion_mapping: FermionMapping::JordanWigner,
            enable_orbital_optimization: true,
            vqe_config: VQEConfig::default(),
        }
    }
}
