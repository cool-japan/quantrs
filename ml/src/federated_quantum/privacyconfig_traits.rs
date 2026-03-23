//! # PrivacyConfig - Trait Implementations
//!
//! This module contains trait implementations for `PrivacyConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::PrivacyConfig;

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            differential_privacy: false,
            privacy_budget: 1.0,
            delta: 1e-5,
            secure_aggregation: false,
            quantum_privacy: Vec::new(),
            data_minimization: true,
        }
    }
}

