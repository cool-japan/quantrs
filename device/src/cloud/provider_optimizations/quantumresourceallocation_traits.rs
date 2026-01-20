//! # QuantumResourceAllocation - Trait Implementations
//!
//! This module contains trait implementations for `QuantumResourceAllocation`.
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

impl Default for QuantumResourceAllocation {
    fn default() -> Self {
        Self {
            qubit_count: 10,
            quantum_volume: None,
            gate_fidelity_requirements: HashMap::new(),
            coherence_time_requirements: CoherenceTimeRequirements {
                min_t1_us: 100.0,
                min_t2_us: 50.0,
                min_gate_time_ns: 100.0,
                thermal_requirements: 0.01,
            },
        }
    }
}
