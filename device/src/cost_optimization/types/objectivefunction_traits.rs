//! # ObjectiveFunction - Trait Implementations
//!
//! This module contains trait implementations for `ObjectiveFunction`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ObjectiveFunction;

impl std::fmt::Debug for ObjectiveFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObjectiveFunction")
            .field("name", &self.name)
            .field("function", &"<function>")
            .field("optimization_direction", &self.optimization_direction)
            .field("weight", &self.weight)
            .finish()
    }
}

impl Clone for ObjectiveFunction {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            function: Box::new(|_| 0.0),
            optimization_direction: self.optimization_direction.clone(),
            weight: self.weight,
        }
    }
}
