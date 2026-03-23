//! # FunctionGuard - Trait Implementations
//!
//! This module contains trait implementations for `FunctionGuard`.
//!
//! ## Implemented Traits
//!
//! - `Drop`
//! - `Send`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "plotters")]
use plotters::prelude::*;

use super::types::FunctionGuard;

impl Drop for FunctionGuard {
    fn drop(&mut self) {
        if let Some(profiler_ptr) = self.profiler {
            unsafe {
                (*profiler_ptr).exit_function(&self.name);
            }
        }
    }
}

unsafe impl Send for FunctionGuard {}
