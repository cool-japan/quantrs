//! # ActiveSpaceConfig - Trait Implementations
//!
//! This module contains trait implementations for `ActiveSpaceConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;

use super::types::{ActiveSpaceConfig, OrbitalSelectionStrategy};

impl Default for ActiveSpaceConfig {
    fn default() -> Self {
        Self {
            active_electrons: 10,
            active_orbitals: 10,
            orbital_selection: OrbitalSelectionStrategy::EnergyBased,
            energy_window: None,
            occupation_threshold: 0.02,
        }
    }
}
