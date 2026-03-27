//! # TensorNetwork - Trait Implementations
//!
//! This module contains trait implementations for `TensorNetwork`.
//!
//! ## Implemented Traits
//!
//! - `Display`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, SimulatorError};
use quantrs2_circuit::prelude::*;
use quantrs2_core::prelude::*;
use std::fmt;

use super::types::TensorNetwork;

impl fmt::Display for TensorNetwork {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "TensorNetwork with {} qubits:", self.num_qubits)?;
        writeln!(f, "  Tensors: {}", self.tensors.len())?;
        writeln!(f, "  Connections: {}", self.connections.len())?;
        writeln!(f, "  Memory usage: {} bytes", self.memory_usage())?;
        Ok(())
    }
}
