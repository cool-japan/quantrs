//! # RetryConfig - Trait Implementations
//!
//! This module contains trait implementations for `RetryConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            backoff_strategy: BackoffStrategy::Exponential,
            retry_conditions: vec![
                RetryCondition::NetworkError,
                RetryCondition::QuantumBackendError,
                RetryCondition::TimeoutError,
            ],
        }
    }
}
