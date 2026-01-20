//! # FiniteDifferenceDifferentiator - Trait Implementations
//!
//! This module contains trait implementations for `FiniteDifferenceDifferentiator`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Differentiator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{MLError, Result};
use crate::simulator_backends::{DynamicCircuit, Observable, SimulationResult, SimulatorBackend};
use quantrs2_circuit::prelude::*;
use quantrs2_core::prelude::*;

use super::functions::Differentiator;
use super::types::FiniteDifferenceDifferentiator;

impl Default for FiniteDifferenceDifferentiator {
    fn default() -> Self {
        Self::new()
    }
}

impl Differentiator for FiniteDifferenceDifferentiator {
    fn differentiate(
        &self,
        circuit: &DynamicCircuit,
        parameters: &[f64],
        observable: &Observable,
        backend: &dyn SimulatorBackend,
    ) -> Result<Vec<f64>> {
        let mut gradients = Vec::with_capacity(parameters.len());
        for i in 0..parameters.len() {
            let mut params_plus = parameters.to_vec();
            params_plus[i] += self.epsilon;
            let exp_plus = backend.expectation_value(circuit, &params_plus, observable)?;
            let mut params_minus = parameters.to_vec();
            params_minus[i] -= self.epsilon;
            let exp_minus = backend.expectation_value(circuit, &params_minus, observable)?;
            let gradient = (exp_plus - exp_minus) / (2.0 * self.epsilon);
            gradients.push(gradient);
        }
        Ok(gradients)
    }
    fn name(&self) -> &str {
        "FiniteDifference"
    }
}
