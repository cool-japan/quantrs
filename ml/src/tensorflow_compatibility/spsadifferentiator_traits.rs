//! # SPSADifferentiator - Trait Implementations
//!
//! This module contains trait implementations for `SPSADifferentiator`.
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
use super::types::SPSADifferentiator;

impl Default for SPSADifferentiator {
    fn default() -> Self {
        Self::new()
    }
}

impl Differentiator for SPSADifferentiator {
    fn differentiate(
        &self,
        circuit: &DynamicCircuit,
        parameters: &[f64],
        observable: &Observable,
        backend: &dyn SimulatorBackend,
    ) -> Result<Vec<f64>> {
        let n_params = parameters.len();
        let mut gradients = vec![0.0; n_params];
        for _ in 0..self.num_samples {
            let delta: Vec<f64> = (0..n_params)
                .map(|_| if fastrand::bool() { 1.0 } else { -1.0 })
                .collect();
            let params_plus: Vec<f64> = parameters
                .iter()
                .zip(delta.iter())
                .map(|(p, d)| p + self.epsilon * d)
                .collect();
            let params_minus: Vec<f64> = parameters
                .iter()
                .zip(delta.iter())
                .map(|(p, d)| p - self.epsilon * d)
                .collect();
            let exp_plus = backend.expectation_value(circuit, &params_plus, observable)?;
            let exp_minus = backend.expectation_value(circuit, &params_minus, observable)?;
            let diff = exp_plus - exp_minus;
            for (i, d) in delta.iter().enumerate() {
                gradients[i] += diff / (2.0 * self.epsilon * d);
            }
        }
        for g in &mut gradients {
            *g /= self.num_samples as f64;
        }
        Ok(gradients)
    }
    fn name(&self) -> &str {
        "SPSA"
    }
}
