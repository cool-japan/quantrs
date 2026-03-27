//! # LQGConfig - Trait Implementations
//!
//! This module contains trait implementations for `LQGConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::random::prelude::*;
use std::f64::consts::PI;

use super::types::LQGConfig;

impl Default for LQGConfig {
    fn default() -> Self {
        Self {
            barbero_immirzi_parameter: 0.2375,
            max_spin: 5.0,
            num_nodes: 100,
            num_edges: 300,
            spin_foam_dynamics: true,
            area_eigenvalues: (1..=20)
                .map(|j| f64::from(j) * (PI * 1.616e-35_f64.powi(2)))
                .collect(),
            volume_eigenvalues: (1..=50)
                .map(|n| f64::from(n).sqrt() * 1.616e-35_f64.powi(3))
                .collect(),
            holonomy_discretization: 0.1,
        }
    }
}
