//! # QuantumGate - Trait Implementations
//!
//! This module contains trait implementations for `QuantumGate`.
//!
//! ## Implemented Traits
//!
//! - `Hash`
//! - `PartialEq`
//! - `Eq`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

#[cfg(feature = "advanced_math")]
use quantrs2_circuit::prelude::*;
use std::hash::{Hash, Hasher};

use super::types::QuantumGate;

impl Hash for QuantumGate {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.gate_type.hash(state);
        self.qubits.hash(state);
        for &param in &self.parameters {
            ((param * 1000.0).round() as i64).hash(state);
        }
    }
}

impl PartialEq for QuantumGate {
    fn eq(&self, other: &Self) -> bool {
        self.gate_type == other.gate_type
            && self.qubits == other.qubits
            && self.parameters.len() == other.parameters.len()
            && self
                .parameters
                .iter()
                .zip(other.parameters.iter())
                .all(|(&a, &b)| (a - b).abs() < 1e-10)
    }
}

impl Eq for QuantumGate {}
