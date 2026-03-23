//! # Constraint - Trait Implementations
//!
//! This module contains trait implementations for `Constraint`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::Constraint;

impl std::fmt::Debug for Constraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Constraint")
            .field("name", &self.name)
            .field("constraint_function", &"<function>")
            .field("constraint_type", &self.constraint_type)
            .field("bound", &self.bound)
            .finish()
    }
}

impl Clone for Constraint {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            constraint_function: Box::new(|_| 0.0),
            constraint_type: self.constraint_type.clone(),
            bound: self.bound,
        }
    }
}
