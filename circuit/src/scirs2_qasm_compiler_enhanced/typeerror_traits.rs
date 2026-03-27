//! # TypeError - Trait Implementations
//!
//! This module contains trait implementations for `TypeError`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::TypeError;
use std::fmt;

impl fmt::Display for TypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Type error: expected {}, found {}",
            self.expected, self.found
        )
    }
}
