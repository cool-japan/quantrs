//! # AutoParallelEngine - generate_gate_dependency_uuid_group Methods
//!
//! This module contains method implementations for `AutoParallelEngine`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use uuid::Uuid;

use super::autoparallelengine_type::AutoParallelEngine;

impl AutoParallelEngine {
    /// Generate a deterministic UUID for a gate index to track dependencies
    pub(super) fn generate_gate_dependency_uuid(gate_index: usize) -> Uuid {
        let namespace =
            Uuid::parse_str("6ba7b810-9dad-11d1-80b4-00c04fd430c8").unwrap_or_else(|_| Uuid::nil());
        let mut bytes = [0u8; 16];
        let index_bytes = gate_index.to_le_bytes();
        bytes[0..8].copy_from_slice(&index_bytes);
        Uuid::from_bytes(bytes)
    }
}
