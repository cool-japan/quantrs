//! # StatisticalAnalysis - Trait Implementations
//!
//! This module contains trait implementations for `StatisticalAnalysis`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;
use std::collections::{BTreeMap, HashMap, VecDeque};

use super::types::{CorrelationMatrix, StatisticalAnalysis};

impl Default for StatisticalAnalysis {
    fn default() -> Self {
        Self {
            suite_statistics: HashMap::new(),
            cross_suite_correlations: CorrelationMatrix::new(),
            significance_tests: Vec::new(),
            confidence_intervals: HashMap::new(),
        }
    }
}
