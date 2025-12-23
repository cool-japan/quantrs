//! # ComputeResourceAllocation - Trait Implementations
//!
//! This module contains trait implementations for `ComputeResourceAllocation`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::traits::ProviderOptimizer;
use super::types::*;
use crate::prelude::CloudProvider;
use crate::DeviceResult;
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

impl Default for ComputeResourceAllocation {
    fn default() -> Self {
        Self {
            cpu_cores: 4,
            memory_gb: 16.0,
            gpu_resources: None,
            specialized_processors: Vec::new(),
        }
    }
}
