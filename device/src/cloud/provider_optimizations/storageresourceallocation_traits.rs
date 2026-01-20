//! # StorageResourceAllocation - Trait Implementations
//!
//! This module contains trait implementations for `StorageResourceAllocation`.
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

impl Default for StorageResourceAllocation {
    fn default() -> Self {
        Self {
            storage_type: StorageType::SSD,
            capacity_gb: 100.0,
            iops_requirements: None,
            throughput_requirements: None,
        }
    }
}
