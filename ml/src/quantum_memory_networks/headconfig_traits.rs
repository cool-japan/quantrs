//! # HeadConfig - Trait Implementations
//!
//! This module contains trait implementations for `HeadConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{HeadConfig, HeadType};

impl Default for HeadConfig {
    fn default() -> Self {
        Self {
            num_read_heads: 4,
            num_write_heads: 2,
            read_head_type: HeadType::QuantumAttention,
            write_head_type: HeadType::QuantumGated,
            entangled_heads: true,
            interaction_strength: 0.8,
        }
    }
}
