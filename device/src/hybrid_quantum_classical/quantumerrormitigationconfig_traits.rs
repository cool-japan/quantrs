//! # Quantumerrormitigationconfig - Trait Implementations
//!
//! This module contains trait implementations for `Quantumerrormitigationconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for QuantumErrorMitigationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strategies: vec![
                ErrorMitigationStrategy::ZeroNoiseExtrapolation,
                ErrorMitigationStrategy::ReadoutErrorMitigation,
            ],
            adaptive_mitigation: true,
            confidence_threshold: 0.95,
        }
    }
}
