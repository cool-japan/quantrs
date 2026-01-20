//! # ComparativeAnalyzer - Trait Implementations
//!
//! This module contains trait implementations for `ComparativeAnalyzer`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;
use std::sync::{Arc, Mutex};

use super::types::{BaselineDatabase, ComparativeAnalyzer};

impl Default for ComparativeAnalyzer {
    fn default() -> Self {
        Self {
            baseline_db: Arc::new(Mutex::new(BaselineDatabase::new())),
        }
    }
}
