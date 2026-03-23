//! # QuantumChemistryDMRGConfig - Trait Implementations
//!
//! This module contains trait implementations for `QuantumChemistryDMRGConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{
    ActiveSpaceConfig, BasisSetType, ElectronicStructureMethod, ExchangeCorrelationFunctional,
    QuantumChemistryDMRGConfig,
};

impl Default for QuantumChemistryDMRGConfig {
    fn default() -> Self {
        Self {
            num_orbitals: 10,
            num_electrons: 10,
            max_bond_dimension: 1000,
            convergence_threshold: 1e-8,
            max_sweeps: 20,
            electronic_method: ElectronicStructureMethod::CASSCF,
            molecular_geometry: Vec::new(),
            basis_set: BasisSetType::STO3G,
            xcfunctional: ExchangeCorrelationFunctional::B3LYP,
            state_averaging: false,
            num_excited_states: 0,
            temperature: 0.0,
            active_space: ActiveSpaceConfig::default(),
            point_group_symmetry: None,
        }
    }
}
