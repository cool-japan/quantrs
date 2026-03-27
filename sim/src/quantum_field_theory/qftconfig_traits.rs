//! # QFTConfig - Trait Implementations
//!
//! This module contains trait implementations for `QFTConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;
use std::collections::{HashMap, VecDeque};

use super::types::{FieldTheoryType, QFTBoundaryConditions, QFTConfig, RenormalizationScheme};

impl Default for QFTConfig {
    fn default() -> Self {
        let mut couplings = HashMap::new();
        couplings.insert("g".to_string(), 0.1);
        couplings.insert("lambda".to_string(), 0.01);
        Self {
            spacetime_dimensions: 4,
            lattice_size: vec![16, 16, 16, 32],
            lattice_spacing: 1.0,
            field_theory: FieldTheoryType::ScalarPhi4,
            boundary_conditions: QFTBoundaryConditions::Periodic,
            temperature: 0.0,
            chemical_potential: 0.0,
            coupling_constants: couplings,
            gauge_invariant: true,
            renormalization_scheme: RenormalizationScheme::DimensionalRegularization,
            mc_steps: 10_000,
        }
    }
}
