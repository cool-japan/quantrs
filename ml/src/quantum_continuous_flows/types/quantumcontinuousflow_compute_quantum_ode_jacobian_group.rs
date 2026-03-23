//! # QuantumContinuousFlow - compute_quantum_ode_jacobian_group Methods
//!
//! This module contains method implementations for `QuantumContinuousFlow`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use scirs2_core::ndarray::*;
use scirs2_core::random::prelude::*;

use super::types::QuantumFlowState;

use super::quantumcontinuousflow_type::QuantumContinuousFlow;

impl QuantumContinuousFlow {
    /// Compute quantum ODE Jacobian determinant
    pub(super) fn compute_quantum_ode_jacobian(
        &self,
        state: &QuantumFlowState,
        integration_time: f64,
    ) -> Result<f64> {
        let trace_estimate = state
            .amplitudes
            .iter()
            .map(|amp| amp.norm_sqr().ln())
            .sum::<f64>();
        Ok(trace_estimate * integration_time)
    }
}
