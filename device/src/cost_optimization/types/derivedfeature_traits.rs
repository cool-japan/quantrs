//! # DerivedFeature - Trait Implementations
//!
//! This module contains trait implementations for `DerivedFeature`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::DerivedFeature;

impl std::fmt::Debug for DerivedFeature {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DerivedFeature")
            .field("name", &self.name)
            .field("computation", &"<function>")
            .field("dependencies", &self.dependencies)
            .finish()
    }
}

impl Clone for DerivedFeature {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            computation: Box::new(|_| 0.0),
            dependencies: self.dependencies.clone(),
        }
    }
}
