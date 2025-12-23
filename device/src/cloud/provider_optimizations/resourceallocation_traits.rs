//! # ResourceAllocation - Trait Implementations
//!
//! This module contains trait implementations for `ResourceAllocation`.
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

impl Default for ResourceAllocation {
    fn default() -> Self {
        Self {
            compute_resources: ComputeResourceAllocation::default(),
            storage_resources: StorageResourceAllocation::default(),
            network_resources: NetworkResourceAllocation::default(),
            quantum_resources: QuantumResourceAllocation::default(),
        }
    }
}
