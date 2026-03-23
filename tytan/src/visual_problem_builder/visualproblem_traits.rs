//! # VisualProblem - Trait Implementations
//!
//! This module contains trait implementations for `VisualProblem`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::VisualProblem;
use std::fmt;

impl Default for VisualProblem {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for VisualProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Problem '{}': {} variables, {} constraints",
            self.metadata.name,
            self.variables.len(),
            self.constraints.len()
        )
    }
}
