//! # SciRS2ParallelContext - Trait Implementations
//!
//! This module contains trait implementations for `SciRS2ParallelContext`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::{
    current_num_threads, IndexedParallelIterator, ParallelIterator, ThreadPool, ThreadPoolBuilder,
};
use scirs2_core::random::prelude::*;

use super::types::SciRS2ParallelContext;

impl Default for SciRS2ParallelContext {
    fn default() -> Self {
        let num_threads = current_num_threads();
        let thread_pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap_or_else(|_| {
                ThreadPoolBuilder::new()
                    .build()
                    .expect("fallback thread pool creation should succeed")
            });
        Self {
            num_threads,
            thread_pool,
            numa_aware: true,
        }
    }
}
