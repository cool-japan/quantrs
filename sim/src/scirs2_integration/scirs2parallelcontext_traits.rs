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
use std::sync::{Arc, OnceLock};

use super::types::SciRS2ParallelContext;

/// Process-wide worker pool backing every [`SciRS2ParallelContext`].
static SHARED_THREAD_POOL: OnceLock<Arc<ThreadPool>> = OnceLock::new();

/// Returns the shared worker pool, building it on first use.
///
/// A `SciRS2ParallelContext` is constructed by every `SciRS2Backend::new()`, and thus by
/// every `StateVectorSimulator::new()`. Building one rayon pool per context spawns
/// `num_threads` OS threads each time, which dominates the runtime of any workload that
/// creates a simulator in a loop (parameter-shift gradients create one per evaluation).
/// One pool for the process keeps that cost at zero after the first call.
pub fn shared_thread_pool() -> Arc<ThreadPool> {
    Arc::clone(SHARED_THREAD_POOL.get_or_init(|| {
        let pool = ThreadPoolBuilder::new()
            .num_threads(current_num_threads())
            .build()
            .unwrap_or_else(|_| {
                ThreadPoolBuilder::new()
                    .build()
                    .expect("fallback thread pool creation should succeed")
            });
        Arc::new(pool)
    }))
}

impl Default for SciRS2ParallelContext {
    fn default() -> Self {
        let thread_pool = shared_thread_pool();
        Self {
            num_threads: thread_pool.current_num_threads(),
            thread_pool,
            numa_aware: true,
        }
    }
}
