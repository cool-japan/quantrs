//! # CommunicationConstraints - Trait Implementations
//!
//! This module contains trait implementations for `CommunicationConstraints`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ClassicalSecurityConfig, CommunicationConstraints, CompressionConfig};

impl Default for CommunicationConstraints {
    fn default() -> Self {
        Self {
            max_comm_rounds: 100,
            bandwidth_limit: 10.0,
            max_latency: 1000.0,
            compression: CompressionConfig {
                enabled: true,
                compression_ratio: 0.5,
                quantization_bits: 8,
                sparsification_threshold: 0.01,
                quantum_compression: None,
            },
            quantum_protocols: Vec::new(),
            classical_security: ClassicalSecurityConfig {
                encryption: "AES-256".to_string(),
                key_length: 256,
                authentication: "HMAC-SHA256".to_string(),
                cert_validation: true,
            },
        }
    }
}

