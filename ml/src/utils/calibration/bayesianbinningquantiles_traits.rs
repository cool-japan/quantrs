//! # BayesianBinningQuantiles - Trait Implementations
//!
//! This module contains trait implementations for `BayesianBinningQuantiles`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::BayesianBinningQuantiles;

impl Default for BayesianBinningQuantiles {
    fn default() -> Self {
        Self::new(10)
    }
}
