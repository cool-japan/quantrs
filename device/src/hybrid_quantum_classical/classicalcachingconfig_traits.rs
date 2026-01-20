//! # Classicalcachingconfig - Trait Implementations
//!
//! This module contains trait implementations for `Classicalcachingconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for ClassicalCachingConfig {
    fn default() -> Self {
        Self {
            enable_caching: true,
            cache_size_mb: 1024.0,
            eviction_policy: CacheEvictionPolicy::LRU,
            persistent_cache: false,
            enable_compression: true,
        }
    }
}
