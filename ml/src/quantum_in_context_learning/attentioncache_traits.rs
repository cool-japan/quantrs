//! # AttentionCache - Trait Implementations
//!
//! This module contains trait implementations for `AttentionCache`.
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
use std::collections::HashMap;
use std::f64::consts::PI;

use super::types::AttentionCache;

impl Default for AttentionCache {
    fn default() -> Self {
        Self {
            cached_queries: HashMap::new(),
            cached_keys: HashMap::new(),
            cached_values: HashMap::new(),
            cache_hit_rate: 0.0,
        }
    }
}
