//! # QuantumReservoirComputer - measure_fidelity_group Methods
//!
//! This module contains method implementations for `QuantumReservoirComputer`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::Result;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;

use super::quantumreservoircomputer_type::QuantumReservoirComputer;

impl QuantumReservoirComputer {
    /// Measure fidelity with reference state
    pub(super) fn measure_fidelity(&self) -> Result<Array1<f64>> {
        let fidelity = self.reservoir_state.state_vector[0].norm_sqr();
        Ok(Array1::from_vec(vec![fidelity]))
    }
}
