//! # Datamanagementconfig - Trait Implementations
//!
//! This module contains trait implementations for `Datamanagementconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for DataManagementConfig {
    fn default() -> Self {
        Self {
            storage_strategy: DataStorageStrategy::InMemory,
            compression: CompressionConfig::default(),
            serialization_format: SerializationFormat::MessagePack,
            retention_policy: DataRetentionPolicy::default(),
        }
    }
}
