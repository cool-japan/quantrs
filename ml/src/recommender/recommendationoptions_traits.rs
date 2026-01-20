//! # RecommendationOptions - Trait Implementations
//!
//! This module contains trait implementations for `RecommendationOptions`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;
use scirs2_core::{Complex32, Complex64};
use std::f64::consts::PI;

use super::types::RecommendationOptions;

impl Default for RecommendationOptions {
    fn default() -> Self {
        Self {
            exclude_seen: true,
            diversify: false,
            diversity_weight: 0.3,
            explain: false,
            business_rules: None,
        }
    }
}
