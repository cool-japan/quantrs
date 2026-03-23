//! # QuantumContinuousFlow - compute_expectation_value_group Methods
//!
//! This module contains method implementations for `QuantumContinuousFlow`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::{Observable, QuantumFlowState};

use super::quantumcontinuousflow_type::QuantumContinuousFlow;

impl QuantumContinuousFlow {
    /// Compute expectation value of observable
    pub(super) fn compute_expectation_value(
        &self,
        observable: &Observable,
        state: &QuantumFlowState,
    ) -> Result<f64> {
        let mut expectation = 0.0;
        for &qubit in &observable.qubits {
            if qubit < state.amplitudes.len() {
                expectation += state.amplitudes[qubit].norm_sqr();
            }
        }
        Ok(expectation)
    }
}
