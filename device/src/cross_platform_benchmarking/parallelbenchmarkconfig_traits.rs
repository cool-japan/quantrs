//! # ParallelBenchmarkConfig - Trait Implementations
//!
//! This module contains trait implementations for `ParallelBenchmarkConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use quantrs2_circuit::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};

use super::types::{LoadBalancingStrategy, ParallelBenchmarkConfig};

impl Default for ParallelBenchmarkConfig {
    fn default() -> Self {
        Self {
            enable_parallel: true,
            max_concurrent: 4,
            load_balancing: LoadBalancingStrategy::ResourceBased,
            resource_allocation: HashMap::new(),
        }
    }
}
