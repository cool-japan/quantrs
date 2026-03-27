//! # StorageConfig - Trait Implementations
//!
//! This module contains trait implementations for `StorageConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(not(feature = "scirs2"))]
use fallback_scirs2::*;
use quantrs2_circuit::prelude::*;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::types::{
    CompressionAlgorithm, CompressionConfig, PersistenceConfig, StorageBackend, StorageConfig,
};

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            realtime_buffer_size: 10000,
            aggregation_intervals: vec![
                Duration::from_secs(60),
                Duration::from_secs(3600),
                Duration::from_secs(86400),
            ],
            compression: CompressionConfig {
                enabled: true,
                algorithm: CompressionAlgorithm::Gzip,
                ratio_threshold: 0.7,
            },
            persistence: PersistenceConfig {
                enabled: true,
                backend: StorageBackend::Memory,
                batch_size: 1000,
                write_interval: Duration::from_secs(60),
            },
        }
    }
}
