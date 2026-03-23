//! # TopologicalConfig - Trait Implementations
//!
//! This module contains trait implementations for `TopologicalConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    AnyonModel, LatticeType, SurfaceCode, TopologicalBoundaryConditions, TopologicalConfig,
    TopologicalErrorCode,
};

impl Default for TopologicalConfig {
    fn default() -> Self {
        Self {
            lattice_type: LatticeType::SquareLattice,
            dimensions: vec![8, 8],
            anyon_model: AnyonModel::Abelian,
            boundary_conditions: TopologicalBoundaryConditions::Periodic,
            temperature: 0.0,
            magnetic_field: 0.1,
            topological_protection: true,
            error_correction_code: TopologicalErrorCode::SurfaceCode,
            enable_braiding: true,
        }
    }
}
