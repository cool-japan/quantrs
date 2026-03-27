//! # ParameterShiftGradient - Trait Implementations
//!
//! This module contains trait implementations for `ParameterShiftGradient`.
//!
//! ## Implemented Traits
//!
//! - `GradientCalculator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::circuit_interfaces::{InterfaceCircuit, InterfaceGate, InterfaceGateType};
use crate::error::{Result, SimulatorError};
use scirs2_core::random::prelude::*;

use super::functions::{CostFunction, GradientCalculator};
use super::types::ParameterShiftGradient;

impl GradientCalculator for ParameterShiftGradient {
    fn calculate_gradient(
        &self,
        parameters: &[f64],
        cost_function: &dyn CostFunction,
        circuit: &InterfaceCircuit,
    ) -> Result<Vec<f64>> {
        let shift = std::f64::consts::PI / 2.0;
        let mut gradient = Vec::with_capacity(parameters.len());
        for i in 0..parameters.len() {
            let mut params_plus = parameters.to_vec();
            let mut params_minus = parameters.to_vec();
            params_plus[i] += shift;
            params_minus[i] -= shift;
            let cost_plus = cost_function.evaluate(&params_plus, circuit)?;
            let cost_minus = cost_function.evaluate(&params_minus, circuit)?;
            let grad = (cost_plus - cost_minus) / 2.0;
            gradient.push(grad);
        }
        Ok(gradient)
    }
    fn method_name(&self) -> &'static str {
        "ParameterShift"
    }
}
