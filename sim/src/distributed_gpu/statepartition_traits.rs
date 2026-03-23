//! # StatePartition - Trait Implementations
//!
//! This module contains trait implementations for `StatePartition`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use scirs2_core::parallel_ops::*;

use super::types::StatePartition;

impl std::fmt::Debug for StatePartition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StatePartition")
            .field("start_index", &self.start_index)
            .field("size", &self.size)
            .field("device_id", &self.device_id)
            .finish()
    }
}
