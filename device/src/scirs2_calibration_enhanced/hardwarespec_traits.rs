//! # HardwareSpec - Trait Implementations
//!
//! This module contains trait implementations for `HardwareSpec`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::parallel_ops::*;
use scirs2_core::random::prelude::*;
use std::collections::{BTreeMap, HashMap, VecDeque};

use super::types::HardwareSpec;

impl Default for HardwareSpec {
    fn default() -> Self {
        Self {
            device_name: "Generic Quantum Device".to_string(),
            num_qubits: 5,
            connectivity: vec![(0, 1), (1, 2), (2, 3), (3, 4)],
            gate_set: vec!["X", "Y", "Z", "H", "S", "T", "CNOT", "CZ"]
                .into_iter()
                .map(String::from)
                .collect(),
            readout_error: 0.01,
            gate_errors: HashMap::new(),
            coherence_times: HashMap::new(),
        }
    }
}
