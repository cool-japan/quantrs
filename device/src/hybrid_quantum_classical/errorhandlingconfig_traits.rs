//! # Errorhandlingconfig - Trait Implementations
//!
//! This module contains trait implementations for `Errorhandlingconfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for ErrorHandlingConfig {
    fn default() -> Self {
        Self {
            recovery_strategies: vec![
                ErrorRecoveryStrategy::Retry,
                ErrorRecoveryStrategy::Fallback,
            ],
            retry_config: RetryConfig::default(),
            fallback_mechanisms: vec![
                FallbackMechanism::AlternativeBackend,
                FallbackMechanism::SimulatorFallback,
            ],
            error_reporting: ErrorReportingConfig::default(),
        }
    }
}
